from typing import Tuple, Union

import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import (ASAPooling, GATConv, GraphConv, RGCNConv,
                                SAGEConv, SAGPooling, TransformerConv,GCNConv)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_sparse import SparseTensor, matmul
from torchtools.optim import RangerLars

from utils import torch_metrics

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


class Dense(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Dense, self).__init__()
        self.bn = nn.BatchNorm1d(outchannel)
        self.act = Mish()
        self.layer = nn.Linear(inchannel,outchannel)
        
    def forward(self,x):
        x = self.act(self.bn(self.layer(x)))
        return x
    
    
class MLP(nn.Module):
    def __init__(self, inchannel, outchannel, Numlayer = 2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList([Dense(inchannel,outchannel)])
        for _ in range(Numlayer-1):
            self.layers.append(Dense(outchannel, outchannel))
        
    def forward(self,x):
        for l in self.layers:
            x = l(x)
        return x

class FocalLoss(nn.Module):
    """ Focal loss for better handling imbalanced class distribution. """
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class UselessConv(MessagePassing):
    def __init__(self,**kwargs):  
        kwargs.setdefault('aggr', 'mean')
        super(UselessConv, self).__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += x_r

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class GNN(nn.Module):
    def __init__(self,inDim, outDim, gnn_type = "Transformer",  *args, **kwargs):
        '''
        Basic GNN model.
        '''
        super(GNN,self).__init__()
        self.gnn = self._getGNN(gnn_type, inDim, outDim)
        self.norm = nn.BatchNorm1d(inDim)
        self.act = Mish()

    def _getGNN(self,gnn_type, inDim, outDim, *args, **kwargs):
        if gnn_type == "GCN":
            return GCNConv(inDim, outDim, *args, **kwargs)
        elif gnn_type == "SAGE":
            return SAGEConv(inDim, outDim, *args, **kwargs)
        elif gnn_type == "GAT":
            return GATConv(inDim, outDim, *args, **kwargs)
        elif gnn_type == "Transformer":
            return TransformerConv(inDim, outDim, heads=2, *args, **kwargs)
        else:
            raise NotImplementedError
        
    def forward(self,x,edge_index):
        feature = self.gnn(x,edge_index)
        feature = self.act(self.norm(feature))
        return feature

class GNNStack(nn.Module):
    def __init__(self,layer_dims,numLayers,*args, **kwargs):
        '''
        Create a model that stacks multi-layer GNN.
        The embedding obtainable are concatenated as output. 
        '''
        super(GNNStack,self).__init__()
        self.gnns = nn.ModuleList([])
        for i in range(numLayers):
            gnn_module = GNN(layer_dims[0], layer_dims[1], *args, **kwargs)
            self.gnns.append(gnn_module)
            
    
    def forward(self,x,adjs):
        final_node = adjs[-1][-1][1]
        features = [x[:final_node]] # raw feature
        current_feature = x
        for i, (edge_index, eid, size) in enumerate(adjs):
            x_target = current_feature[:size[1]]  # Target nodes are always placed first.
            current_feature = self.gnns[i]((current_feature, x_target), edge_index)
            current_feature = F.dropout(current_feature, p=0.5, training=self.training)
            features.append(current_feature[:final_node])
    
        return features 
    
class LabelAwarePool(nn.Module):
    def __init__(self,ratio = 0.5, nonlinear = nn.Sigmoid(), distance = "L1"):
        super().__init__()
        self.nonlinear = nonlinear
        self.dis = distance
        self.ratio = ratio 
        
    def forward(self,logits, edge_index):
        logits = self.nonlinear(logits).flatten()
        row, col = logits[edge_index]
        pair_disntance = self.distance(row,col)
        similarity = 1 - pair_disntance
        new_edges = []
        source_nodes = edge_index[1]
        for source_idx in range(source_nodes.max()+1):
            source_edges = source_nodes == source_idx
            source_simi = similarity[source_edges]
            source_edges = edge_index[:,source_edges]
            topK_edgeIndex = torch.topk(source_simi, k = int(source_simi.size(0) * self.ratio)).indices
            selected_edge = source_edges[:,topK_edgeIndex]
            new_edges.append(selected_edge)
        new_edges = torch.cat(new_edges,dim=-1)    
        
        return new_edges

    def distance(self,x,y):
        if self.dis == "L1":
            dis = torch.abs(x-y)
        elif self.dis == "L2":
            dis = (x-y)**2    
        return dis
    
    
class GNNSelector(nn.Module):
    def __init__(self,layer_dims,numLayers):
        '''
        Create a model that stacks multi-layer GNN.
        The embedding obtainable are concatenated as output. 
        '''
        super(GNNSelector,self).__init__()
        self.pool = nn.ModuleList([LabelAwarePool(ratio=0.5), LabelAwarePool(ratio=1)])
        self.gnns = nn.ModuleList([])
        self.outLayers = nn.ModuleList([nn.Linear(layer_dims[0], 1)])
        for i in range(numLayers):
            gnn_module = GNN(layer_dims[0], layer_dims[1])
            self.gnns.append(gnn_module)
            self.outLayers.append(nn.Linear(layer_dims[0], 1))
    
    def forward(self,x,adjs):
        # init logits
        Target_logits = []
        final_node = adjs[-1][-1][1]
        current_feature = x
        current_logits = self.outLayers[0](current_feature)
        Target_logits.append(current_logits[:final_node])
        
        # getting logits and new edges
        for i, (edge_index, eid, size) in enumerate(adjs):
            new_edge = self.pool[i](current_logits, edge_index)
            x_target = current_feature[:size[1]]  # Target nodes are always placed first.
            current_feature = self.gnns[i]((current_feature, x_target), new_edge)
            current_logits = self.outLayers[i+1](current_feature)
            current_feature = F.dropout(current_feature, p=0.5, training=self.training)
            Target_logits.append(current_logits[:final_node])
    
        return Target_logits 



class SSLGNN(LightningModule):
    def __init__(self, data, input_dim, hidden_dim, numLayers, useXGB=True):
        super().__init__()
        self.save_hyperparameters()
        self.data = data
        self.input_dim = input_dim
        self.dim = hidden_dim*2
        self.numLayers = numLayers
        self.layers = [self.dim, self.dim//2] #* (numLayers+1)
        self.bn = nn.BatchNorm1d(self.dim)
        self.act = Mish()
        self.useXGB = useXGB
        
        # GNN layer
        if self.useXGB:
            self.initEmbedding = nn.Embedding(self.input_dim, self.dim, padding_idx=0)
        else:
            self.initEmbedding = MLP(self.input_dim, self.dim, Numlayer=2)
        self.initGNN = UselessConv()
        self.GNNs = GNNStack(self.layers,self.numLayers)
        
        # output
        self.outLayer = nn.ModuleList([nn.Linear(self.dim,1) for _ in range(numLayers+1)])
        self.revLayer = nn.ModuleList([nn.Linear(self.dim,1) for _ in range(numLayers+1)])
        self.combined = nn.Linear(numLayers+1, 1, bias=False)
        self.combinedRev = nn.Linear(numLayers+1, 1, bias=False)
        self.loss_func = FocalLoss(logits=True)

    def forward(self, x,adjs):
        # update node embedding
        leaf_emb = self.initEmbedding(x)
        if self.useXGB:
            leaf_emb = torch.sum(leaf_emb,dim=1) # summation over the trees
            leaf_emb = self.bn(leaf_emb)
            leaf_emb = self.act(leaf_emb)
        
        # first update 
        firstHop_neighbor = adjs[-2][0]
        leaf_emb = self.initGNN(leaf_emb,to_undirected(firstHop_neighbor))
        
        # GNN 
        embeddings = self.GNNs(leaf_emb, adjs)
        
        # logits
        logits = [self.outLayer[i](v) for i,v in enumerate(embeddings)]
        ensemble = torch.cat(logits, dim=-1)
        ensemble = self.combined(ensemble)
        logits.append(ensemble)
        
        # revenue
        revenues = [torch.relu(self.revLayer[i](v)) for i,v in enumerate(embeddings)]
        ensemble = torch.cat(revenues, dim=-1)
        ensemble = self.combinedRev(ensemble)
        revenues.append(ensemble)
        
        return logits, revenues
    
    def compute_CLS_loss(self,Logits, label):
        loss = 0
        for logit in Logits:
            logit = logit.flatten()
            l = self.loss_func(logit,label)
            loss+= l
        return loss
    
    def compute_REG_loss(self,preds, rev):
        loss = 0
        for pred in preds:
            pred = pred.flatten()
            l = F.mse_loss(pred,rev)
            loss += l
        return loss 

    def training_step(self, batch, batch_idx: int):
        target_idx = torch.arange(batch.y.shape[0])
        logits, revenues = self(batch.x, batch.adjs_t)
        CLS_loss = self.compute_CLS_loss(logits, batch.y)  
        REG_loss = self.compute_REG_loss(revenues, batch.rev)
        train_loss = CLS_loss + 10 * REG_loss
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        target_idx = torch.arange(batch.y.shape[0])
        logits, revenues = self(batch.x, batch.adjs_t)
        CLS_loss = self.compute_CLS_loss(logits, batch.y)  
        REG_loss = self.compute_REG_loss(revenues, batch.rev)
        valid_loss = CLS_loss + 1 * REG_loss
        self.log('val_loss', valid_loss, on_step=True, on_epoch=True, sync_dist=True)
        return logits[-1]
    
    def validation_epoch_end(self, val_step_outputs):
        predictions = torch.cat(val_step_outputs).detach().cpu().numpy().ravel()
        f,pr, re, rev = torch_metrics(predictions, self.data.valid_cls_label, self.data.valid_reg_label)
        f1_top = np.mean(f)
        self.log("F1-top",f1_top)
        performance = [*f, *pr, *re, *rev]
        name_performance = ["F1@1","F1@2","F1@5","F1@10","Pr@1","Pr@2","Pr@5","Pr@10",
                            "Re@1","Re@2","Re@5","Re@10","Rev@1","Rev@2","Rev@5","Rev@10"]
        name_performance = ["Val/"+i for i in name_performance]
        tensorboard_logs = dict(zip(name_performance,performance))
        return {"Val/F1-top":f1_top, "log":tensorboard_logs}
        
    def test_step(self,batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self,val_step_outputs):
        predictions = torch.cat(val_step_outputs).detach().cpu().numpy().ravel()
        f,pr, re, rev = torch_metrics(predictions, self.data.test_cls_label, self.data.test_reg_label)
        f1_top = np.mean(f)
        performance = [*f, *pr, *re, *rev]
        name_performance = ["F1@1","F1@2","F1@5","F1@10","Pr@1","Pr@2","Pr@5","Pr@10",
                            "Re@1","Re@2","Re@5","Re@10","Rev@1","Rev@2","Rev@5","Rev@10"]
        name_performance = ["Test/"+i for i in name_performance]
        tensorboard_logs = dict(zip(name_performance,performance))
        
        return {"F1-top":f1_top, "log":tensorboard_logs}

    def configure_optimizers(self):
        optimizer = RangerLars(self.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
        return [optimizer], [scheduler]

    
class HeteroGNN(nn.Module):
    def __init__(self,inDim, outDim, n_relations):
        '''
        Basic GNN model.
        '''
        super(HeteroGNN,self).__init__()
        self.gnn = RGCNConv(inDim, outDim,n_relations)
        self.norm = nn.BatchNorm1d(outDim)
        self.act = Mish()
        
    def forward(self,x,edge_index, edge_type):
        feature = self.gnn(x,edge_index,edge_type)
        feature = self.act(self.norm(feature))
        return feature
    
    
class HeteroGNNStack(nn.Module):
    def __init__(self,layer_dims,numLayers,n_relations,*args, **kwargs):
        '''
        Create a model that stacks multi-layer GNN.
        The embedding obtainable are concatenated as output. 
        '''
        super(HeteroGNNStack,self).__init__()
        self.gnns = nn.ModuleList([])
        for i in range(numLayers):
            gnn_module = HeteroGNN(layer_dims[0], layer_dims[1],n_relations)
            self.gnns.append(gnn_module)
            
    
    def forward(self,x,adjs, edge_attr):
        final_node = adjs[-1][-1][1]
        features = [x[:final_node]] # raw feature
        current_feature = x
        for i, (edge_index, eid, size) in enumerate(adjs):
            edge_type = edge_attr[eid]
            current_feature = self.gnns[i](current_feature, edge_index, edge_type)
            current_feature = F.dropout(current_feature, p=0.5, training=self.training)
            features.append(current_feature[:final_node])
        return features