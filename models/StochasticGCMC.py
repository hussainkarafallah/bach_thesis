import torch as th
import torch.nn as nn
import torch.nn.functional as thF
import dgl.function as gF
from tqdm import tqdm
import numpy as np
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization

import commons

class StochasticGCMC(GeneralRecommender):

    input_type = InputType.PAIRWISE
    __name__ = 'S-GCMC'
    default_params = {
        'gcn_output_dim' : 500,
        'embedding_size' : 64,
        'node_dropout' : 0.2,
        'dense_dropout' : 0.2,
        'fans' : [None],
        'train_batch_size' : 512,
    }
    def __init__(self, config, dataset):
        super(StochasticGCMC, self).__init__(config, dataset)

        # load parameters info
        gcn_output_dim = config['gcn_output_dim']
        embedding_size = config['embedding_size']
        self.node_dropout = nn.Dropout(config['node_dropout'])
        self.dense_dropout = nn.Dropout(config['dense_dropout'])
        self.fans = config['fans']
        self.num_layers = 1

        # load dataset info
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.cpu_graph = dataset.graph
        self.embedding = th.nn.Embedding(self.cpu_graph.number_of_nodes() , gcn_output_dim)
        self.check_point = th.zeros( (self.cpu_graph.number_of_nodes() , embedding_size)).to(commons.device)


        assert min(self.cpu_graph.in_degrees()) > 0

        self.loss = nn.CrossEntropyLoss()

        # parameters initialization
        self.W1 = nn.Linear(gcn_output_dim , embedding_size)
        self.W2 = nn.Linear(gcn_output_dim , embedding_size)
        self.decoder = DenseBiDecoder(embedding_size , 2)
        self.reset_parameters()

    def reset_parameters(self):
        from torch.nn.init import xavier_uniform_
        xavier_uniform_(self.W1.weight)
        xavier_uniform_(self.W2.weight)
        xavier_uniform_(self.embedding.weight)
        self.decoder.reset_parameters()

    @staticmethod
    def msg(edges):
        return {
            'm' : edges.src['norm'] * edges.src['x'] * edges.dst['norm']
        }

    def forward_block(self , block , h):
        with block.local_scope():
            block.srcdata['x'] = self.node_dropout(h)
            block.update_all(self.msg , gF.sum('m' , 'y'))
            x = thF.relu(block.dstdata['y'])
            x = self.dense_dropout(x)
            return x


    def forward(self , batch):
        users, pos, neg, blocks = batch
        split = users.max() + 1
        h_users = self.W1(self.embedding(blocks[0].srcdata['_ID'][:split]))
        h_items = self.W2(self.embedding(blocks[0].srcdata['_ID'][split:]))
        h = th.cat((h_users , h_items) , dim = 0)
        for block in blocks:
            h = self.forward_block(block, h)

        users, pos, neg = h[users], h[pos], h[neg]
        predictions = self.decoder(th.cat((users, users)), th.cat((pos, neg)))
        return predictions

    def calculate_loss(self, batch):
        users, pos, neg, blocks = batch
        predictions = self.forward(batch)
        target = th.zeros(len(users) * 2, dtype=th.long).to(self.device)
        target[:len(users)] = 1
        return self.loss(predictions , target)

    @th.no_grad()
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID] + self.num_users
        user , item = self.check_point[user] , self.check_point[item]
        ret = self.decoder(user , item)[:,1]
        return ret.cpu()

    @th.no_grad()
    def inference(self, mode='validation', verbose = False):
        assert mode in ['validation' , 'testing'] , "got mode {}".format(mode)
        from dgl.dataloading import NodeDataLoader , MultiLayerNeighborSampler
        self.eval()
        if mode == 'testing':
            sampler = MultiLayerNeighborSampler([None])
        else:
            sampler = MultiLayerNeighborSampler(self.fans)
        g = self.cpu_graph
        kwargs = {
            'batch_size' : 64,
            'shuffle' : True,
            'drop_last' : False,
            'num_workers' : 6,
        }
        dataloader = NodeDataLoader(g,th.arange(g.number_of_nodes()), sampler,**kwargs)
        if verbose:
            dataloader = tqdm(dataloader)

        x = self.embedding.weight
        x = th.cat((self.W1(x[:self.num_users]), self.W2(x[self.num_users:])), dim=0)


        # Within a layer, iterate over nodes in batches
        for input_nodes, output_nodes, blocks in dataloader:
            block = blocks[0].to(commons.device)
            h = self.forward_block(block , x[input_nodes])
            self.check_point[output_nodes] = h

        if verbose:
            print('Inference Done Successfully')


class DenseBiDecoder(nn.Module):
    def __init__(self,in_units,num_classes,num_basis=2,dropout_rate=0.0):
        super().__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ufeat, ifeat):
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = th.einsum('ai,bij,aj->ab', ufeat, self.P, ifeat)
        out = self.combine_basis(out)
        return out