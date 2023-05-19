import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.pool.topk_pool import filter_adj
from torch_geometric.nn import Set2Set, global_sort_pool
from torch_geometric.utils import to_dense_batch
from utils import topk, generate_sub_head_nums
import torch.nn.functional as F
from layers import SAGPool

class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class NetGlobal(torch.nn.Module):
    def __init__(self, args):
        super(NetGlobal, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool = SAGPool(self.nhid * 3, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 6, self.nhid * 2)
        self.lin2 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([x], dim=-1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([x], dim=-1)

        x = F.relu(self.conv3(x, edge_index))
        x = torch.cat([x1, x2, x], dim=-1)
        x, edge_index, _, batch, _ = self.pool(x, edge_index, None, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

# class MultiHeadLayer(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, head_num, pooling_rate, GCN_act=F.relu, att_pooling='max',
#             pooling_nonlinearity=torch.tanh):
#         super(MultiHeadLayer, self).__init__()
#
#         self.head_num = head_num
#         self.sub_head_dims = generate_sub_head_nums(input_dim, head_num)
#         self.head_num = len(self.sub_head_dims)
#         if hidden_dim % self.head_num != 0:
#             raise Exception('The hidden dimension of each block should be times of head number! '
#                             'Current head number is %d and hidden dimension is %d', self.head_num, hidden_dim)
#         self.sub_hidden_dim = int(hidden_dim / self.head_num)
#         GCNs = []
#         for i in range(self.head_num):
#             GCNs.append(GCNConv(self.sub_head_dims[i], self.sub_hidden_dim))
#         self.convs = torch.nn.ModuleList(GCNs)
#         self.atts = torch.nn.ModuleList([torch.nn.Linear(self.sub_hidden_dim, 1, bias=False) for i in range(self.head_num)])
#         if att_pooling == 'concat':
#             self.pooling_score_layer = GCNConv(self.sub_hidden_dim * self.head_num, 1)
#         else:
#             self.pooling_score_layer = GCNConv(self.sub_hidden_dim, 1)
#         self.pooling_nonlinearity = pooling_nonlinearity
#         self.pooling_rate = pooling_rate
#         self.GCN_act = GCN_act
#         self.att_pooling = att_pooling
#
#     def forward(self, x, edge_index, batch, edge_attr=None):
#         if self.head_num != 1:
#             sub_xs = []
#             start = 0
#             for sub_dim in self.sub_head_dims:
#                 sub_xs.append(x[:, start: start + sub_dim])
#         else:
#             sub_xs = [x]
#         att_results = []
#         sub_xs_out = []
#         for i in range(self.head_num):
#             x = self.GCN_act(self.convs[i](sub_xs[i], edge_index))
#             sub_xs_out.append(x)
#             att_score = self.atts[i](x)
#             att_results.append((att_score * x))
#
#         if self.att_pooling == 'concat':
#             att_pooling_result = torch.cat(att_results, -1)
#         else:
#             for i in range(len(att_results)):
#                 att_results[i] = att_results[i].unsqueeze(-1)
#             att_results = torch.cat(att_results, -1)
#             if self.att_pooling == 'max':
#                 att_pooling_result = att_results.max(-1)[0]
#             elif self.att_pooling == 'mean':
#                 att_pooling_result = att_results.mean(-1)
#             elif self.att_pooling == 'min':
#                 att_pooling_result = att_results.min(-1)[0]
#
#         score = self.pooling_score_layer(att_pooling_result, edge_index).squeeze()
#         if len(score.size()) == 0:
#             score = score.unsqueeze(0)
#         perm = topk(score, self.pooling_rate, batch)
#         x = torch.cat(sub_xs_out, -1)
#         x = x[perm] * self.pooling_nonlinearity(score[perm]).view(-1, 1)
#         batch = batch[perm]
#         try:
#             edge_index, edge_attr = filter_adj(
#                 edge_index, edge_attr, perm, num_nodes=score.size(0))
#         except:
#             print('111')
#         return x, edge_index, edge_attr, batch, perm

class MultiHeadAttNew(torch.nn.Module):
    def __init__(self, args, head_num, sub_hidden_dim, pooling_rate, att_type='standard', GCN_act=F.relu,
                 pooling_nonlinearity=torch.tanh):
        super(MultiHeadAttNew, self).__init__()
        self.head_num = head_num
        self.sub_hidden_dim = sub_hidden_dim
        self.att_pooling = args.att_pooling_type
        self.att_type = args.att_type
        self.att_weight_type = args.att_weight_type
        self.att_type = att_type
        self.score_type = args.score_type
        self.ablation = args.ablation
        if self.ablation == 'mlpatt':
            self.atts = torch.nn.ModuleList(
                [torch.nn.Linear(self.sub_hidden_dim, 1) for i in range(self.head_num)])
        else:
            self.atts = torch.nn.ModuleList(
                [GCNConv(self.sub_hidden_dim, 1) for i in range(self.head_num)])
        if self.ablation == 'mlpscore':
            self.pooling_score_layer = torch.nn.Linear(1, 1)
        else:
            self.pooling_score_layer = GCNConv(1, 1)
        self.pooling_nonlinearity = pooling_nonlinearity
        self.pooling_rate = pooling_rate
        self.GCN_act = GCN_act
        if self.att_pooling == 'linear':
            self.att_pooling_linear = torch.nn.Linear(head_num, 1, bias=False)

    def forward(self, x, sub_xs_out, edge_index, batch, edge_attr=None):
        att_results = []
        for i in range(self.head_num):
            if self.ablation == 'mlpatt':
                att_score = F.softmax(self.atts[i](sub_xs_out[i]))
            else:
                att_score = F.softmax(self.atts[i](sub_xs_out[i], edge_index))
            att_results.append(att_score)

        if self.att_pooling == 'concat':
            att_pooling_result = torch.cat(att_results, -1)
        else:
            for i in range(len(att_results)):
                att_results[i] = att_results[i].unsqueeze(-1)
            if self.att_pooling == 'max':
                att_pooling_result = torch.cat(att_results, -1).max(-1)[0]
            elif self.att_pooling == 'mean':
                att_pooling_result = torch.cat(att_results, -1).mean(-1)
            elif self.att_pooling == 'min':
                att_pooling_result = torch.cat(att_results, -1).min(-1)[0]
            elif self.att_pooling == 'linear':
                att_pooling_result = self.att_pooling_linear(torch.cat(att_results, -1).squeeze(1))
            for i in range(len(att_results)):
                att_results[i] = att_results[i].squeeze(-1)

        if self.ablation == 'noscore':
            score = att_pooling_result.squeeze()
        else:
            if self.ablation == 'mlpscore':
                score = self.pooling_score_layer(att_pooling_result).squeeze()
            else:
                score = self.pooling_score_layer(att_pooling_result, edge_index).squeeze()
        if len(score.size()) == 0:
            score = score.unsqueeze(0)
        if self.ablation == 'nofilter':
            if self.att_weight_type == 'global':
                x = x * self.pooling_nonlinearity(score).view(-1, 1)
            elif self.att_weight_type == 'local':
                for i in range(self.head_num):
                    sub_xs_out[i] = sub_xs_out[i] * att_results[i]
                x = torch.cat(sub_xs_out, -1)
            perm = None
        else:
            perm = topk(score, self.pooling_rate, batch)
            if self.att_weight_type == 'global':
                x = x[perm] * self.pooling_nonlinearity(score[perm]).view(-1, 1)
            elif self.att_weight_type == 'local':
                for i in range(self.head_num):
                    sub_xs_out[i] = sub_xs_out[i][perm] * att_results[i][perm]
                x = torch.cat(sub_xs_out, -1)
            batch = batch[perm]
            edge_index, edge_attr = filter_adj(
                edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm

class MultiHeadAtt(torch.nn.Module):
    def __init__(self, args, head_num, sub_hidden_dim, pooling_rate, att_type='standard', GCN_act=F.relu,
                 pooling_nonlinearity=torch.tanh):
        super(MultiHeadAtt, self).__init__()
        self.head_num = head_num
        self.sub_hidden_dim = sub_hidden_dim
        self.att_pooling = args.att_pooling_type
        self.att_type = args.att_type
        self.att_weight_type = args.att_weight_type
        self.att_type = att_type
        self.score_type = args.score_type
        if att_type == 'standard':
            self.atts = torch.nn.ModuleList(
                [torch.nn.Linear(self.sub_hidden_dim, 1, bias=False) for i in range(self.head_num)])
            if self.att_pooling == 'concat':
                if self.score_type == 'single':
                    self.pooling_score_layer = GCNConv(self.head_num, 1)
                else:
                    self.pooling_score_layer = GCNConv(self.sub_hidden_dim * self.head_num, 1)
            else:
                if self.score_type == 'single':
                    self.pooling_score_layer = GCNConv(1, 1)
                else:
                    self.pooling_score_layer = GCNConv(self.sub_hidden_dim, 1)
        else:
            self.atts = torch.nn.ModuleList(
                [torch.nn.Linear(self.sub_hidden_dim, 1, bias=False) for i in range(self.head_num)])
        self.pooling_nonlinearity = pooling_nonlinearity
        self.pooling_rate = pooling_rate
        self.GCN_act = GCN_act

    def forward(self, x, sub_xs_out, edge_index, batch, edge_attr=None):
        att_results = []
        for i in range(self.head_num):
            att_score = self.atts[i](sub_xs_out[i])
            if self.att_type == 'standard':
                if self.score_type == 'complex':
                    att_results.append((att_score * sub_xs_out[i]))
                elif self.score_type == 'single':
                    att_results.append(att_score)
            else:
                att_results.append(att_score)

        if self.att_pooling == 'concat':
            att_pooling_result = torch.cat(att_results, -1)
        else:
            for i in range(len(att_results)):
                att_results[i] = att_results[i].unsqueeze(-1)
            if self.att_pooling == 'max':
                att_pooling_result = torch.cat(att_results, -1).max(-1)[0]
            elif self.att_pooling == 'mean':
                att_pooling_result = torch.cat(att_results, -1).mean(-1)
            elif self.att_pooling == 'min':
                att_pooling_result = torch.cat(att_results, -1).min(-1)[0]
            for i in range(len(att_results)):
                att_results[i] = att_results[i].squeeze(-1)

        if self.att_type == 'standard':
            score = self.pooling_score_layer(att_pooling_result, edge_index).squeeze()
        else:
            score = att_pooling_result.squeeze()
        if len(score.size()) == 0:
            score = score.unsqueeze(0)
        perm = topk(score, self.pooling_rate, batch)
        if self.att_weight_type == 'global':
            x = x[perm] * self.pooling_nonlinearity(score[perm]).view(-1, 1)
        elif self.att_weight_type == 'local':
            for i in range(self.head_num):
                sub_xs_out[i] = sub_xs_out[i][perm] * self.pooling_nonlinearity(att_results[i][perm])
            x = torch.cat(sub_xs_out, -1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))
        return x, edge_index, edge_attr, batch, perm

class MAGPoolGCNLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, head_num, GCN_act=F.relu):
        super(MAGPoolGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sub_head_dims = generate_sub_head_nums(input_dim, head_num)
        self.head_num = len(self.sub_head_dims)
        if hidden_dim % self.head_num != 0:
            raise Exception('The hidden dimension of each block should be times of head number! '
                            'Current head number is %d and hidden dimension is %d', self.head_num, hidden_dim)
        self.sub_hidden_dim = int(hidden_dim / self.head_num)
        GCNs = []
        for i in range(self.head_num):
            GCNs.append(GCNConv(self.sub_head_dims[i], self.sub_hidden_dim))
        self.convs = torch.nn.ModuleList(GCNs)
        self.GCN_act = GCN_act

    def forward(self, x, edge_index):
        if self.head_num != 1:
            sub_xs = []
            start = 0
            for sub_dim in self.sub_head_dims:
                sub_xs.append(x[:, start: start + sub_dim])
        else:
            sub_xs = [x]
        sub_xs_out = []
        for i in range(self.head_num):
            x = self.GCN_act(self.convs[i](sub_xs[i], edge_index))
            sub_xs_out.append(x)
        x = torch.cat(sub_xs_out, -1)
        return x, sub_xs_out

class MAGPoolGCN(torch.nn.Module):
    def __init__(self, args):
        super(MAGPoolGCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.nhid
        self.num_classes = args.num_classes
        self.pooling_rate = args.pooling_ratio
        self.dropout = args.dropout_ratio
        self.head_num = args.head_num
        self.att_type = args.att_type
        self.multihead_gcn1 = MAGPoolGCNLayer(self.num_features, self.hidden_dim, self.head_num)
        self.multihead_att1 = MultiHeadAtt(args, self.multihead_gcn1.head_num, self.multihead_gcn1.sub_hidden_dim,
                                          self.pooling_rate, att_type=self.att_type)
        self.multihead_gcn2 = MAGPoolGCNLayer(self.hidden_dim, self.hidden_dim, self.head_num)
        self.multihead_att2 = MultiHeadAtt(args, self.multihead_gcn2.head_num, self.multihead_gcn2.sub_hidden_dim,
                                          self.pooling_rate, att_type=self.att_type)
        self.multihead_gcn3 = MAGPoolGCNLayer(self.hidden_dim, self.hidden_dim, self.head_num)
        self.multihead_att3 = MultiHeadAtt(args, self.multihead_gcn3.head_num, self.multihead_gcn3.sub_hidden_dim,
                                          self.pooling_rate, att_type=self.att_type)

        self.linear1 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = torch.nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, sub_xs_out = self.multihead_gcn1(x, edge_index)
        x, edge_index, _, batch, _ = self.multihead_att1(x, sub_xs_out, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, sub_xs_out = self.multihead_gcn2(x, edge_index)
        x, edge_index, _, batch, _ = self.multihead_att2(x, sub_xs_out, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, sub_xs_out = self.multihead_gcn3(x, edge_index)
        x, edge_index, _, batch, _ = self.multihead_att3(x, sub_xs_out, edge_index, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=-1)

        return x

class MAGPoolGCNNew(torch.nn.Module):
    def __init__(self, args):
        super(MAGPoolGCNNew, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.nhid
        self.num_classes = args.num_classes
        self.pooling_rate = args.pooling_ratio
        self.dropout = args.dropout_ratio
        self.head_num = args.head_num
        self.att_type = args.att_type
        self.ablation = args.ablation
        self.multihead_gcn1 = MAGPoolGCNLayer(self.num_features, self.hidden_dim, self.head_num)
        self.multihead_att1 = MultiHeadAttNew(args, self.multihead_gcn1.head_num, self.multihead_gcn1.sub_hidden_dim,
                                          self.pooling_rate, att_type=self.att_type)
        self.multihead_gcn2 = MAGPoolGCNLayer(self.hidden_dim, self.hidden_dim, self.head_num)
        self.multihead_att2 = MultiHeadAttNew(args, self.multihead_gcn2.head_num, self.multihead_gcn2.sub_hidden_dim,
                                          self.pooling_rate, att_type=self.att_type)
        self.multihead_gcn3 = MAGPoolGCNLayer(self.hidden_dim, self.hidden_dim, self.head_num)
        self.multihead_att3 = MultiHeadAttNew(args, self.multihead_gcn3.head_num, self.multihead_gcn3.sub_hidden_dim,
                                          self.pooling_rate, att_type=self.att_type)

        self.linear1 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = torch.nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, sub_xs_out = self.multihead_gcn1(x, edge_index)
        x, edge_index, _, batch, _ = self.multihead_att1(x, sub_xs_out, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, sub_xs_out = self.multihead_gcn2(x, edge_index)
        x, edge_index, _, batch, _ = self.multihead_att2(x, sub_xs_out, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, sub_xs_out = self.multihead_gcn3(x, edge_index)
        x, edge_index, _, batch, _ = self.multihead_att3(x, sub_xs_out, edge_index, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if self.ablation == 'noreadout':
            x = x3
        else:
            x = x1 + x2 + x3

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=-1)

        return x

class MAGPoolGCNGlobal(torch.nn.Module):
    def __init__(self, args):
        super(MAGPoolGCNGlobal, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.nhid
        self.num_classes = args.num_classes
        self.pooling_rate = args.pooling_ratio
        self.dropout = args.dropout_ratio
        self.ablation = args.ablation
        self.multihead_gcn1 = MAGPoolGCNLayer(self.num_features, self.hidden_dim, 2)
        self.multihead_gcn2 = MAGPoolGCNLayer(self.hidden_dim, self.hidden_dim, 2)
        self.multihead_gcn3 = MAGPoolGCNLayer(self.hidden_dim, self.hidden_dim, 2)
        self.multihead_att = MultiHeadAttNew(args, 3, self.hidden_dim, self.pooling_rate)

        self.linear1 = torch.nn.Linear(self.hidden_dim * 6, self.hidden_dim * 2)
        self.linear2 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear3 = torch.nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, _ = self.multihead_gcn1(x, edge_index)
        x1 = x.clone().detach()

        x, _ = self.multihead_gcn2(x, edge_index)
        x2 = x.clone().detach()

        x, _ = self.multihead_gcn3(x, edge_index)
        x3 = x.clone().detach()

        if self.ablation == 'noreadout':
            x = x3
        else:
            x =  x = torch.cat([x1, x2, x3], -1)

        x, edge_index, _, batch, _ = self.multihead_att(x, [x1, x2, x3], edge_index, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=-1)

        return x

class Set2SetNet(torch.nn.Module):
    def __init__(self, args):
        super(Set2SetNet, self).__init__()
        self.hidden_dim = args.num_features
        self.num_classes = args.num_classes
        self.dropout = args.dropout_ratio
        self.set2set = Set2Set(self.hidden_dim, 1)

        self.linear1 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = torch.nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.set2set(x, batch)
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.log_softmax(self.linear3(x), dim=-1)

        return x

class SortPoolNet(torch.nn.Module):
    def __init__(self, args):
        super(SortPoolNet, self).__init__()
        self.hidden_dim = args.num_features
        self.num_classes = args.num_classes
        self.dropout = args.dropout_ratio
        self.gcn1 = GCNConv(self.hidden_dim, 32)
        self.gcn2 = GCNConv(32, 32)
        self.gcn3 = GCNConv(32, 32)
        self.gcn4 = GCNConv(32, 1)
        self.k = args.sort_k # COLLAB: 34, Mutagenicity: 14

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=self.k,
                            out_channels=16,
                            kernel_size=2,
                            stride=2),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16,
                            out_channels=32,
                            kernel_size=5,
                            stride=1),
            torch.nn.ReLU()
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, self.num_classes))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.tanh(self.gcn1(x, edge_index))
        x2 = F.tanh(self.gcn2(x1, edge_index))
        x3 = F.tanh(self.gcn3(x2, edge_index))
        score = F.tanh(self.gcn4(x3, edge_index))
        x = torch.cat([x1, x2, x3], -1)

        fill_value = score.min().item() - 1
        batch_x = to_dense_batch(x, batch, fill_value)[0].squeeze()
        B, N, D = batch_x.size()

        batch_score, _ = to_dense_batch(score, batch, fill_value)
        _, perm = batch_score[:, :, -1].sort(dim=-1, descending=True)
        arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
        perm = perm + arange.view(-1, 1)

        batch_x = batch_x.view(B * N, D)
        batch_x = batch_x[perm]
        batch_x = batch_x.view(B, N, D)
        if N >= self.k:
            x = batch_x[:, :self.k].contiguous()
        else:
            expand_batch_x = batch_x.new_full((B, self.k - N, D), fill_value)
            x = torch.cat([batch_x, expand_batch_x], dim=1)

        x = self.conv1(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool1d(x, 5, 1)
        x = x.view(B, -1)
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)

        return x
