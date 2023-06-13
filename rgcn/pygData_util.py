from typing import List, NamedTuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch import Tensor
from torch_geometric.data import Data,NeighborSampler
from torch_sparse import SparseTensor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

import dataset
from utils import process_leaf_idx


class GraphData(object):
    def __init__(self,data, categories = ["HS6","importer.id"], use_xgb=True, pos_weight = 2):
        '''
        Generate dataset in pytorch-geometric format
        This class extract bipartite graph that links transaction to target categories
        Args:
            data (Custuom dataset): customs dataset obtainable from dataset.py
            categories (List(str)): list of target categories for bipartite graph
            use_xgb (bool): if True, use pre-trained XGB multi-hot vector as input. otherwise the raw feature is used
        '''
        self.data = data
        self.node_num = 0 
        self.categories = categories
        self.G = nx.Graph()
        
        # xgb config
        self.use_xgb = use_xgb
        self.num_trees = 100
        self.depth = 4
        self.pos_weight = pos_weight
        
        # nodeid mapping
        self.n2id = dict()
        self.id2n = dict()
        if self.use_xgb:
            self.train_xgb_model()
        else:
            self.prepare_df()
        self.prepare_DATE_input()
        
    def train_xgb_model(self):
        ''' 
        Train XGB model to obtain leaf indices
        '''
        print("Training XGBoost model...")
        self.xgb = XGBClassifier(n_estimators=self.num_trees, max_depth=self.depth, n_jobs=-1, eval_metric="error", scale_pos_weight = self.pos_weight)
#         self.xgb = GradientBoostingClassifier(n_estimators=self.num_trees, max_depth=self.depth)
        self.xgb.fit(self.data.dftrainx_lab, self.data.train_cls_label)   
    
        # Get leaf index from xgboost model 
        X_train_leaves = self.xgb.apply(self.data.dftrainx_lab)
        X_trainunlab_leaves = self.xgb.apply(self.data.dftrainx_unlab)
        X_valid_leaves = self.xgb.apply(self.data.dfvalidx_lab)
        X_test_leaves = self.xgb.apply(self.data.dftestx)
        
        # One-hot encoding for leaf index
        X_leaves = np.concatenate((X_train_leaves, X_trainunlab_leaves, X_valid_leaves, X_test_leaves), axis=0)
        transformed_leaves, self.leaf_dim, new_leaf_index = process_leaf_idx(X_leaves)
        train_rows = X_train_leaves.shape[0]
        trainunlab_rows = X_trainunlab_leaves.shape[0] + train_rows
        valid_rows = X_valid_leaves.shape[0] + trainunlab_rows
        self.train_leaves, self.trainunlab_leaves, self.valid_leaves, self.test_leaves = transformed_leaves[:train_rows],\
                                          transformed_leaves[train_rows:trainunlab_rows],\
                                          transformed_leaves[trainunlab_rows:valid_rows],\
                                          transformed_leaves[valid_rows:]
        
    def prepare_df(self):
        '''
        Normalize input DataFrame to (0,1) to train NN
        '''
        train_data = pd.concat((self.data.dftrainx_lab,self.data.dftrainx_unlab))
        self.leaf_dim = train_data.shape[1]
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_data)
        self.train_leaves = self.scaler.transform(self.data.dftrainx_lab)
        self.trainunlab_leaves = self.scaler.transform(self.data.dftrainx_unlab)
        self.valid_leaves = self.scaler.transform(self.data.dfvalidx_lab)
        self.test_leaves = self.scaler.transform(self.data.dftestx)

    def prepare_DATE_input(self):
        """ Prepare input for Dual-Attentive Tree-Aware Embedding model, DATE """
                
        # user & item information 
        train_raw_importers = self.data.train_lab['importer.id'].values
        train_raw_items = self.data.train_lab['tariff.code'].values
        valid_raw_importers = self.data.valid_lab['importer.id'].values
        valid_raw_items = self.data.valid_lab['tariff.code'].values
        test_raw_importers = self.data.test['importer.id']
        test_raw_items = self.data.test['tariff.code']

        # we need padding for unseen user or item 
        importer_set = set(train_raw_importers) 
        item_set = set(train_raw_items) 

        # Remember to +1 for zero padding 
        importer_mapping = {v:i+1 for i,v in enumerate(importer_set)} 
        hs6_mapping = {v:i+1 for i,v in enumerate(item_set)}
        self.importer_size = len(importer_mapping) + 1
        self.item_size = len(hs6_mapping) + 1
        train_importers = [importer_mapping[x] for x in train_raw_importers]
        train_items = [hs6_mapping[x] for x in train_raw_items]

        # for test data, we use padding_idx=0 for unseen data
        valid_importers = [importer_mapping.get(x,0) for x in valid_raw_importers]
        valid_items = [hs6_mapping.get(x,0) for x in valid_raw_items]
        test_importers = [importer_mapping.get(x,0) for x in test_raw_importers] # use dic.get(key,deafault) to handle unseen
        test_items = [hs6_mapping.get(x,0) for x in test_raw_items]

        # Convert to torch type
        self.train_user = torch.tensor(train_importers).long()
        self.train_item = torch.tensor(train_items).long()
        self.valid_user = torch.tensor(valid_importers).long()
        self.valid_item = torch.tensor(valid_items).long()
        self.test_user = torch.tensor(test_importers).long()
        self.test_item = torch.tensor(test_items).long()
        
        
    def _getDF(self,stage):
        if stage == "train_lab":
            raw_df = self.data.train_lab
            feature = torch.LongTensor(self.train_leaves) if self.use_xgb else torch.FloatTensor(self.train_leaves)
            sideFeature = [self.train_user, self.train_item]
            
        elif stage =="train_unlab":
            raw_df = self.data.train_unlab
            feature = torch.LongTensor(self.trainunlab_leaves) if self.use_xgb else torch.FloatTensor(self.trainunlab_leaves)
            sideFeature = [torch.zeros(feature.shape[0]).long(), torch.zeros(feature.shape[0]).long()]
            
        elif stage == "valid":
            raw_df = self.data.valid_lab
            feature = torch.LongTensor(self.valid_leaves) if self.use_xgb else torch.FloatTensor(self.valid_leaves)
            sideFeature = [self.valid_user, self.valid_item]
            
        elif stage == "test":
            raw_df = self.data.test
            feature = torch.LongTensor(self.test_leaves) if self.use_xgb else torch.FloatTensor(self.test_leaves)
            sideFeature = [self.test_user, self.test_item]
        else:
            raise KeyError("No such stage for building dataframe")
        return raw_df, feature, sideFeature
    
    def _getNid(self,x):
        '''
        Return node indice from a given id
        If the target id is not in current graph, create a new node for it
        Args:
            x : identifier of a node
        '''
        # get node index from raw data
        if x in self.id2n.keys():
            return self.id2n[x]
        else:
            self.id2n[x] = self.node_num
            self.n2id[self.node_num] = x
            self.node_num += 1
            return self.node_num - 1
        
    def _get_revenue(self,stage):
        if stage == "train_lab":
            return torch.FloatTensor(self.data.norm_revenue_train)
        elif stage =="train_unlab":
            return torch.ones(self.data.train_unlab.shape[0]).float()
        elif stage == "valid":
            return torch.FloatTensor(self.data.norm_revenue_valid)
        elif stage == "test":
            return torch.FloatTensor(self.data.norm_revenue_test)
        
    def get_AttNode(self,stage):
        '''Get all node id for a certain stage'''
        nodes = [x for x,y in self.G.nodes(data=True) if y["att"]==stage]
        return nodes
    
    def get_data(self,stage):
        '''
        obtain pyG data for a certain stage. 
        The data contains
            x: node feature
            y: binary target for classification. 
               The label of category nodes are initialized as -1
            rev: additional revenue for dual task learning
            edge_index: edges of the  graph
            edge_att: type of edge denotes the relation between transaction and category
            edge_label: 1 if one of node in the edge is illicit else 0  (not used)
        '''
        pyg_data = Data()
        edges = []
        edge_att = []
        edge_label = []
        df, node_feature,  sideFeature = self._getDF(stage)
        importers = sideFeature[0]
        items = sideFeature[1]
        transaction_nodes = [self._getNid(i) for i in df.index]
        self.G.add_nodes_from(transaction_nodes, att = stage)
        target = torch.FloatTensor(df["illicit"].values)
        rev_target = self._get_revenue(stage)
        current_nodeNum = self.node_num
        for cid, cvalue in enumerate(self.categories):
            for gid, groups in df.groupby(cvalue):
                transaction_ids = list(groups.index)
                transaction_nodeid = [self._getNid(i) for i in transaction_ids]
                categoryNid = self._getNid(str(cvalue)+str(gid)) # convert to string incase duplication with transaction id
                
                # add node attribute
                self.G.add_node(categoryNid, att = cvalue)
                
                # create edges
                current_edges = list(zip(transaction_nodeid, [categoryNid] * len(transaction_nodeid)))
                self.G.add_edges_from(current_edges)
                edge_type = [cid] * len(transaction_nodeid)
                edge_target = groups["illicit"].values.tolist()
                edges.extend(current_edges)
                edge_att.extend(edge_type)
                edge_label.extend(edge_target)
                
        # append node feature (for categories)
        new_nodeNum = self.node_num - current_nodeNum
        init_feature = torch.zeros(new_nodeNum,self.num_trees) if self.use_xgb else torch.zeros(new_nodeNum,self.leaf_dim)
        init_feature = init_feature.long() if self.use_xgb else init_feature
        node_feature = torch.cat((node_feature,init_feature), dim=0)
        importers = torch.cat((importers,torch.zeros(new_nodeNum).long()))
        items = torch.cat((items,torch.zeros(new_nodeNum).long()))
        target = torch.cat((target, -torch.ones(new_nodeNum)))
        rev_target = torch.cat((rev_target, -torch.ones(new_nodeNum)))
                
        # PyG data format
        pyg_data.x = node_feature
        pyg_data.y = target
        pyg_data.rev = rev_target
        pyg_data.edge_index = torch.LongTensor(edges).T
        pyg_data.edge_index = torch.cat((pyg_data.edge_index, torch.flip(pyg_data.edge_index,[0])), dim=-1)
        pyg_data.edge_attr = torch.LongTensor(edge_att+edge_att)
        pyg_data.edge_label = torch.FloatTensor(edge_label + edge_label)
        pyg_data.importer = importers
        pyg_data.item = items
        
        return pyg_data



def StackData(train_data, unlab_data, valid_data, test_data):
    '''
    stack pyG dataset.
    because the valid/test data should include train/unlab edges
    '''
    stack = Data()
    all_data = (train_data, unlab_data, valid_data, test_data)
    
    # feature
    x = [data.x for data in all_data]
    x = torch.cat(x,dim=0)
    stack.x = x

    # importer
    importer = [data.importer for data in all_data]
    importer = torch.cat(importer,dim=-1)
    stack.importer = importer

    # item 
    item = [data.item for data in all_data]
    item = torch.cat(item,dim=-1)
    stack.item = item
    
    # target
    y = [data.y for data in all_data]
    y = torch.cat(y,dim=-1)
    stack.y = y
    
    # revenue
    rev = [data.rev for data in all_data]
    rev = torch.cat(rev,dim=-1)
    stack.rev = rev
    
    # edge type
    edge_attr = [data.edge_attr for data in all_data]
    edge_attr = torch.cat(edge_attr,dim=-1)
    stack.edge_attr = edge_attr
    
    # edge index
    stack.train_edge = torch.cat((train_data.edge_index, unlab_data.edge_index), dim=1)
    stack.valid_edge = torch.cat((stack.train_edge,valid_data.edge_index ), dim=1)
    stack.test_edge = torch.cat((stack.valid_edge,test_data.edge_index ), dim=1)
    
    # transaction index
    stack.train_idx = train_data.node_idx
    stack.valid_idx = valid_data.node_idx
    stack.test_idx = test_data.node_idx
    
    return stack


class Batch(NamedTuple):
    '''
    convert batch data for pytorch-lightning
    '''
    x: Tensor
    y: Tensor
    rev: Tensor
    adjs_t: NamedTuple
    importer: Tensor
    item : Tensor

    def to(self, *args, **kwargs):
        return Batch(
            x=self.x.to(*args, **kwargs),
            y=self.y.to(*args, **kwargs),
            rev=self.rev.to(*args, **kwargs),
            adjs_t=[(adj_t.to(*args, **kwargs), eid.to(*args, **kwargs), size) for adj_t, eid, size in self.adjs_t],
            importer = self.importer.to(*args, **kwargs),
            item = self.item.to(*args, **kwargs),
        )


class CustomData(LightningDataModule):
    def __init__(self,data,sizes, batch_size = 128):
        '''
        defining dataloader with NeighborSampler to extract k-hop subgraph.
        Args:
            data (Graphdata): graph data for the edges and node index
            sizes ([int]): The number of neighbors to sample for each node in each layer. 
                           If set to :obj:`sizes[l] = -1`, all neighbors are included
            batch_size (int): batch size for training
        '''
        super(CustomData,self).__init__()
        self.data = data
        self.sizes = sizes
        self.valid_sizes = [-1 for i in self.sizes]
        self.batch_size = batch_size

    def train_dataloader(self):
        return NeighborSampler(self.data.train_edge, node_idx=self.data.train_idx,
                               sizes=self.sizes, return_e_id=True,
                               batch_size=self.batch_size,transform=self.convert_batch,
                               shuffle=True,
                               )

    def val_dataloader(self):
        return NeighborSampler(self.data.valid_edge, node_idx=self.data.valid_idx,
                               sizes=self.sizes, return_e_id=True,
                               batch_size=self.batch_size,transform=self.convert_batch,shuffle=False
                              )

    def test_dataloader(self):
        return NeighborSampler(self.data.test_edge, node_idx=self.data.test_idx,
                               sizes=self.sizes, return_e_id=True,
                               batch_size=self.batch_size,transform=self.convert_batch,shuffle=False
                              )

    def convert_batch(self, batch_size, n_id, adjs):
        return Batch(
            x=self.data.x[n_id],
            y=self.data.y[n_id[:batch_size]],
            rev = self.data.rev[n_id[:batch_size]],
            adjs_t=adjs,
            importer = self.data.importer[n_id[:batch_size]],
            item = self.data.item[n_id[:batch_size]],
        )