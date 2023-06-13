import datetime
import random
import os
from collections import defaultdict
from datetime import timedelta
from parser import get_parser

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelEncoder
from torch_cluster import random_walk
from torch_geometric.nn import GATConv,TransformerConv
from pygData_util import *
from torch_geometric.data import Data, DataLoader, Dataset,NeighborSampler
from torch_geometric.utils import from_networkx, to_undirected
from torchtools.optim import RangerLars
from tqdm import tqdm, tqdm_notebook, trange
from xgboost import XGBClassifier

import dataset
from models import *
from models import MLP, GNNStack, Mish, UselessConv
from pygData_util import *
from utils import *

# load data
parser = get_parser()
args = parser.parse_args()
print(args)

chosen_data = args.data
log_name = "%sData" % chosen_data[-1]
if chosen_data == 'real-n':
    data = dataset.Ndata(path='/data1/roytsai/Custom-Semi-Supervised/data/ndata.csv')
elif chosen_data == 'real-m':
    data = dataset.Mdata(path='/data1/roytsai/Custom-Semi-Supervised/data/mdata.csv')
elif chosen_data == 'real-t':
    data = dataset.Tdata(path='/data1/roytsai/Custom-Semi-Supervised/data/tdata.csv')
elif chosen_data == 'real-c':
    data = dataset.Cdata(path='/data1/roytsai/Custom-Semi-Supervised/data/cdata.csv')

# args
seed = args.seed
epochs = args.epoch
batch_size = args.batch_size
dim = args.dim
lr = args.lr
weight_decay = args.l2
initial_inspection_rate = args.initial_inspection_rate
train_begin = args.train_from 
test_begin = args.test_from
test_length = args.test_length
valid_length = args.valid_length
chosen_data = args.data
gpu_id = args.device
pos_weight = args.pos_weight

# Initial dataset split
seed_everything(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

assert initial_inspection_rate < 100, "for supervised setting, please set 99.9 instead of 100"
assert initial_inspection_rate > 0,   "initial_inspection_rate should be greater than 0"

# Initial dataset split
train_start_day = datetime.date(int(train_begin[:4]), int(train_begin[4:6]), int(train_begin[6:8]))
test_start_day = datetime.date(int(test_begin[:4]), int(test_begin[4:6]), int(test_begin[6:8]))
test_length = timedelta(days=test_length)    
test_end_day = test_start_day + test_length
valid_length = timedelta(days=valid_length)
valid_start_day = test_start_day - valid_length

# data
data.split(train_start_day, valid_start_day, test_start_day, test_end_day, valid_length, test_length, args)
data.featureEngineering()

# prepare data
categories=["importer.id","HS6"]
gdata = GraphData(data,use_xgb=True, categories=categories,pos_weight=pos_weight)

stage = "train_lab"
trainLab_data = gdata.get_data(stage)
train_nodeidx = torch.tensor(gdata.get_AttNode(stage))
trainLab_data.node_idx = train_nodeidx

stage = "train_unlab"
unlab_data = gdata.get_data(stage)
unlab_nodeidx = torch.tensor(gdata.get_AttNode(stage))
unlab_data.node_idx = unlab_nodeidx

stage = "valid"
valid_data = gdata.get_data(stage)
valid_nodeidx = torch.tensor(gdata.get_AttNode(stage))
valid_data.node_idx = valid_nodeidx

stage = "test"
test_data = gdata.get_data(stage)
test_nodeidx = torch.tensor(gdata.get_AttNode(stage))
test_data.node_idx = test_nodeidx

# sampler for unsupervised training
class UnsupSampler(NeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(UnsupSampler, self).sample(batch)

class Batch(NamedTuple):
    '''
    convert batch data for pytorch-lightning
    '''
    x: Tensor
    y: Tensor
    rev: Tensor
    adjs_t: NamedTuple
    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            rev=self.rev.to(*args, **kwargs),
            adjs_t=[(adj_t.to(*args, **kwargs), eid.to(*args, **kwargs), size) for adj_t, eid, size in self.adjs_t],
        )

class UnsupData(LightningDataModule):
    def __init__(self,data,sizes, batch_size = 128):
        '''
        defining dataloader with NeighborSampler to extract k-hop subgraph.
        Args:
            data (Graphdata): graph data for the edges and node index
            sizes ([int]): The number of neighbors to sample for each node in each layer. 
                           If set to :obj:`sizes[l] = -1`, all neighbors are included
            batch_size (int): batch size for training
        '''
        super(UnsupData,self).__init__()
        self.data = data
        self.sizes = sizes
        self.valid_sizes = [-1 for i in self.sizes]
        self.batch_size = batch_size

    def train_dataloader(self):
        return UnsupSampler(self.data.train_edge, sizes=self.sizes,
                               batch_size=self.batch_size,transform=self.convert_batch,
                               shuffle=True,num_workers=8)
    
    def test_dataloader(self):
        return UnsupSampler(self.data.test_edge, sizes=self.sizes,node_idx=self.data.test_idx,
                               batch_size=self.batch_size,transform=self.convert_batch,
                               shuffle=False,num_workers=8)
    
    def label_loader(self):
        return UnsupSampler(self.data.test_edge, sizes=self.sizes,node_idx=self.data.train_idx,
                               batch_size=self.batch_size,transform=self.convert_batch,
                               shuffle=False,num_workers=8)

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            rev = self.data.rev[n_id[:batch_size]],
            adjs_t=adjs,
        )

class PretrainGNN(LightningModule):
    def __init__(self,input_dim, hidden_dim, numLayers, n_relations, useXGB=True):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.dim = hidden_dim*2
        self.numLayers = numLayers
        self.layers = [self.dim, self.dim] 
        self.bn = nn.BatchNorm1d(self.dim)
        self.act = Mish()
        self.useXGB = useXGB
        
        # GNN layer
        if self.useXGB:
            self.initEmbedding = nn.Embedding(self.input_dim, self.dim, padding_idx=0)
        else:
            self.initEmbedding = MLP(self.input_dim, self.dim, Numlayer=2)
        self.initGNN = UselessConv()
        self.GNNs = HeteroGNNStack(self.layers,self.numLayers,n_relations)

    def forward(self, x,adjs):
        # update node embedding
        leaf_emb = self.initEmbedding(x)
        if self.useXGB:
            leaf_emb = torch.sum(leaf_emb,dim=1) # summation over the trees
            leaf_emb = self.bn(leaf_emb)
            leaf_emb = self.act(leaf_emb)
        
        # first update 
        firstHop_neighbor = adjs[-1][0]
        leaf_emb = self.initGNN(leaf_emb,to_undirected(firstHop_neighbor))
        
        # GNN 
        embeddings = self.GNNs(leaf_emb, adjs,edge_attr)
        
        return embeddings[-1]
    
    def training_step(self, batch, batch_idx: int):
        out = self(batch.x, batch.adjs_t)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        train_loss = -pos_loss - neg_loss
        self.log('train_loss', train_loss)
        return train_loss
    
    def test_step(self, batch, batch_idx: int):
        out = self(batch.x, batch.adjs_t)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        return out
    
    def test_epoch_end(self, val_step_outputs):
        val_step_outputs = torch.cat(val_step_outputs)
        val_step_outputs = val_step_outputs.cpu().detach().numpy()
        return {"log":{"predictions":val_step_outputs}}
    
    def configure_optimizers(self):
        optimizer = RangerLars(self.parameters(), lr=0.005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
        return [optimizer], [scheduler]

# pretraining model config
device = "cuda:%s" % gpu_id
input_dim = gdata.leaf_dim
hidden_size = dim
sizes = [50,20]
numLayers = len(sizes)
n_relations = len(gdata.categories)

model = PretrainGNN(input_dim, hidden_size, numLayers, n_relations,useXGB=gdata.use_xgb)

# lightning config
stacked_data = StackData(trainLab_data,unlab_data,valid_data, test_data)
edge_attr = stacked_data.edge_attr.to(device)
datamodule = UnsupData(stacked_data, sizes = sizes, batch_size=batch_size)
trainer = Trainer(gpus=[gpu_id], max_epochs=args.pretrainstep)
trainer.fit(model, train_dataloader=datamodule.train_dataloader())

# model
class Predictor(LightningModule):
    def __init__(self,input_dim, hidden_dim, numLayers, n_relations, useXGB=True):
        super().__init__()
        self.gnn_encoder = PretrainGNN(input_dim, hidden_size, numLayers, n_relations, useXGB)
        self.dim = hidden_dim * 2
        
        # output
        self.clsLayer = nn.Linear(self.dim,1) #GATConv(self.dim,1)
        self.revLayer = nn.Linear(self.dim,1) #GATConv(self.dim,1)
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([1])) #FocalLoss(logits=True)
        
    def load_fromPretrain(self,path):
        self.gnn_encoder.load_from_checkpoint(path)
        
    def loadGNN_state(self,model):
        self.gnn_encoder.load_state_dict(model.state_dict())

    def forward(self, x,adjs):
#         firstHop_neighbor = adjs[-1][0]
        # get node embedding from pre-trained model
        embedding = self.gnn_encoder(x,adjs)
        logit = self.clsLayer(embedding)
        revenue = self.revLayer(embedding)
        
        return logit, revenue
    
    def compute_CLS_loss(self,logit, label):
        logit = logit.flatten()
        loss = self.loss_func(logit,label)
        return loss
    
    def compute_REG_loss(self,pred_rev, rev):
        pred_rev = pred_rev.flatten()
        loss = F.mse_loss(pred_rev,rev)
        return loss 

    def training_step(self, batch, batch_idx: int):
        logits, revenues = self(batch.x, batch.adjs_t)
        CLS_loss = self.compute_CLS_loss(logits, batch.y)  
        REG_loss = self.compute_REG_loss(revenues, batch.rev)
        train_loss = CLS_loss + 10 * REG_loss
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx: int):
        logits, revenues = self(batch.x, batch.adjs_t)
        CLS_loss = self.compute_CLS_loss(logits, batch.y)  
        REG_loss = self.compute_REG_loss(revenues, batch.rev)
        valid_loss = CLS_loss + 1 * REG_loss
        self.log('val_loss', valid_loss, on_step=True, on_epoch=True, sync_dist=True)
        return logits
    
    def validation_epoch_end(self, val_step_outputs):
        predictions = torch.cat(val_step_outputs).detach().cpu().numpy().ravel()
        overall_f1, auc,f,pr, re, rev = metrics(predictions, self.data.valid_cls_label, self.data.valid_reg_label,None)
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
        # self.save_prediction(predictions)
        overall_f1, auc,f,pr, re, rev = metrics(predictions, self.data.test_cls_label, self.data.test_reg_label,None)
        f1_top = np.mean(f)
        performance = [*f, *pr, *re, *rev]
        name_performance = ["F1@1","F1@2","F1@5","F1@10","Pr@1","Pr@2","Pr@5","Pr@10",
                            "Re@1","Re@2","Re@5","Re@10","Rev@1","Rev@2","Rev@5","Rev@10"]
        name_performance = ["Test/"+i for i in name_performance]
        tensorboard_logs = dict(zip(name_performance,performance))
        tensorboard_logs["auc"] = auc
        tensorboard_logs["overall_F1"] = overall_f1

        # open file
        df_summary = pd.DataFrame(tensorboard_logs,index=[0])
        df_summary = df_summary[[i for i in df_summary.columns if "Test" in i]]
        os.makedirs("results", exist_ok=True)
        fpath = "./results/RGCN-OSR-hs-%s-%s.csv" % (log_name,initial_inspection_rate)
        try:
            exp_df = pd.read_csv(fpath)
            exp_df = pd.concat((exp_df,df_summary),axis=0)
            exp_df.to_csv(fpath,index=False)
        except: 
            df_summary.to_csv(fpath,index=False)
        
        return {"F1-top":f1_top, "log":tensorboard_logs}

    def configure_optimizers(self):
        optimizer = RangerLars(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)
        return [optimizer], [scheduler]
    
    def save_prediction(self,predictions):
        ''' saving prediction for analysis'''
        os.makedirs("predictions", exist_ok=True)
        pred_ser = pd.Series(predictions)
        pred_fname = "predictions/GNN_%s_%s_prediction.csv" % (chosen_data,initial_inspection_rate)
        pred_ser.to_csv(pred_fname,index=0)



# model config
input_dim = gdata.leaf_dim
hidden_size = dim
sizes = [-1,200]
numLayers = len(sizes)
predictor = Predictor(input_dim, hidden_size, numLayers, n_relations)
predictor.loadGNN_state(model)
# predictor.load_fromPretrain(pretrain_path)
predictor.data = data

# lightning config
stacked_data = StackData(trainLab_data,unlab_data,valid_data, test_data)
datamodule = CustomData(stacked_data, sizes = sizes, batch_size=batch_size)
logger = TensorBoardLogger("ssl_exp",name=log_name)
logger.log_hyperparams(model.hparams, metrics={"F1-top":0})
checkpoint_callback = ModelCheckpoint(
    monitor='F1-top',    
    dirpath='./saved_model',
    filename='Analysis-%s-{F1-top:.4f}' % log_name,
    save_top_k=1,
    mode='max',
)            
trainer = Trainer(gpus=[gpu_id], max_epochs=epochs,
                 num_sanity_val_steps=0,
                  check_val_every_n_epoch=1,
                  callbacks=[checkpoint_callback],
                 )
trainer.fit(predictor, datamodule=datamodule)
test_summary = trainer.test()

# transform summary as dataframe for saving csv
df_summary = pd.DataFrame(test_summary)
df_summary = df_summary[[i for i in df_summary.columns if "Test" in i]]

# # open file
# os.makedirs("results", exist_ok=True)
# fpath = "./results/GNN-result-%s-%s.csv" % (log_name,initial_inspection_rate)
# try:
#     exp_df = pd.read_csv(fpath)
#     exp_df = pd.concat((exp_df,df_summary),axis=0)
#     exp_df.to_csv(fpath,index=False)
# except: 
#     df_summary.to_csv(fpath,index=False)
# if not os.path.isfile(fpath):
    
# else:
#     exp_df = pd.read_csv(fpath)
#     exp_df = pd.concat((exp_df,df_summary),axis=0)
#     exp_df.to_csv(fpath,index=False)
