import torch.nn as nn
import dgl
import torch as th
import torch.nn.functional as thF
import dgl.function as GF
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from tqdm import tqdm
import commons

leaky_relu = nn.LeakyReLU(0.2)
initializer = nn.init.xavier_uniform_

class NGCFLayer(nn.Module):

    def __init__(self , in_dim , out_dim , config):
        super(NGCFLayer, self).__init__()

        self.msg_dropout = nn.Dropout(config['message_dropout'])
        self.node_dropout = nn.Dropout(config['node_dropout'])

        self.W1 = nn.Linear(in_dim , out_dim)
        self.W2 = nn.Linear(in_dim , out_dim)

    def reset_parameters(self):
        initializer(self.W1.weight)
        initializer(self.W2.weight)

    @staticmethod
    def edge_sum(edges):
        return {
            'm1' : edges.data['m1'] * edges.data['coef'],
            'm2' : edges.data['m2'] * edges.data['coef']
        }

    def pass_messages(self , g):
        g.apply_edges(GF.u_mul_v('norm' , 'norm' , 'coef'))
        g.apply_edges(GF.u_mul_v('x' , 'x' , 'm2'))
        g.apply_edges(GF.copy_u('x' , 'm1'))
        g.apply_edges(self.edge_sum)
        g.update_all(GF.copy_e('m1' , 'm1') , GF.sum('m1' , 'f1'))
        g.update_all(GF.copy_e('m2' , 'm2') , GF.sum('m2' , 'f2'))


    def forward_block(self , block , h):
        with block.local_scope():
            block.srcdata['norm'] = self.node_dropout(block.srcdata['norm'])
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata['x'] = h_src
            block.dstdata['x'] = h_dst
            self.pass_messages(block)
            ret = self.W1(block.dstdata['f1']) + self.W2(block.dstdata['f2'])
            return ret


    def forward_graph(self , graph , h):
        with graph.local_scope():
            graph.ndata['norm'] = self.node_dropout(graph.ndata['norm'])
            graph.ndata['x'] = h
            self.pass_messages(graph)
            ret = self.W1(graph.ndata['f1']) + self.W2(graph.ndata['f2'])
            return ret


class StochasticNGCF(GeneralRecommender):
    input_type = InputType.PAIRWISE
    __name__ = 'S-NGCF'
    default_params = {
        'embedding_size': 64,
        'node_dropout': 0.2,
        'message_dropout': 0.2,
        'fans': None,
        'hidden_size_list' : [64 , 64 , 64],
        'reg_weight': 1e-5,
        'train_batch_size': 2048,

    }

    def __init__(self , config, dataset):
        super(StochasticNGCF, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.node_dropout = nn.Dropout(config['node_dropout'])
        self.msg_dropout = nn.Dropout(config['message_dropout'])

        self.layers_dim = config['hidden_size_list']
        self.num_layers = len(self.layers_dim)
        self.fans = config['fans']
        if not self.fans:
            self.fans = [None] * self.num_layers
        self.decay = config['reg_weight']


        self.layers = []
        sz_arr = [self.embedding_dim] + self.layers_dim
        for i in range(len(self.layers_dim)):
            self.layers.append(NGCFLayer(sz_arr[i] , sz_arr[i+1] , config))
        self.layers = nn.ModuleList(self.layers)

        self.cpu_graph = dataset.graph
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.embedding = nn.Embedding(self.cpu_graph.num_nodes() , self.embedding_dim)
        self.check_point = th.zeros( (self.cpu_graph.num_nodes() , self.embedding_dim * (self.num_layers + 1)) , requires_grad=False).to(commons.device)
        self.reset_parameters()


    def reset_parameters(self):
        initializer(self.embedding.weight)
        for x in self.layers:
            x.reset_parameters()

    def forward_blocks(self, blocks , users, pos_items, neg_items):
        assert len(blocks) == len(self.layers) , "{} , {}".format(blocks , self.layers)
        x = self.embedding(blocks[0].srcdata['_ID'])
        all_embeddings = [x]
        for k , (block , ngcf) in enumerate(zip(blocks , self.layers)):
            ret = ngcf.forward_block(block , all_embeddings[-1][:block.number_of_src_nodes()])
            ret = self.msg_dropout(ret)
            all_embeddings.append(thF.normalize(ret , p = 2 , dim = 1))

        ret_users = th.cat([ arr[users] for arr in all_embeddings] , dim = 1)
        ret_pos = th.cat([ arr[pos_items] for arr in all_embeddings] , dim = 1)
        ret_neg = th.cat([ arr[neg_items] for arr in all_embeddings] , dim = 1)

        return ret_users , ret_pos , ret_neg

    def calculate_loss(self, batch):
        users, pos, neg, blocks = batch
        users_e , pos_e , neg_e = self.forward_blocks(blocks , users , pos , neg)
        return self.create_bpr_loss(users_e , pos_e , neg_e)[0]

    @th.no_grad()
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID] + self.num_users
        user, item = self.check_point[user], self.check_point[item]
        ret = th.mul(user , item).sum(dim = 1)
        return ret.cpu()

    @th.no_grad()
    def inference(self, mode='validation', verbose = False):
        assert mode in ['validation', 'testing'], "got mode {}".format(mode)
        from dgl.dataloading import NodeDataLoader, MultiLayerNeighborSampler
        self.eval()
        if mode == 'testing':
            sampler = MultiLayerNeighborSampler([None] * self.num_layers)
        else:
            sampler = MultiLayerNeighborSampler(self.fans)

        g = self.cpu_graph
        kwargs = {
            'batch_size': 1024,
            'shuffle': True,
            'drop_last': False,
            'num_workers': commons.workers,
        }

        dataloader = NodeDataLoader(g, th.arange(g.number_of_nodes()), sampler, **kwargs)
        # Within a layer, iterate over nodes in batches
        if verbose:
            dataloader = tqdm(dataloader)
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [ x.to(commons.device) for x in blocks ]
            users = th.arange(output_nodes.shape[0]).long().to(self.device)
            d1 = th.zeros((0,)).long().to(self.device)
            d2 = th.zeros((0,)).long().to(self.device)
            h = self.forward_blocks(blocks , users , d1 , d2)[0]
            self.check_point[output_nodes] = h
        if verbose:
            print('Inference Done Successfully')

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embedding = self.check_point[user]
        i_embeddings = self.check_point[self.num_users : ]
        assert i_embeddings.shape[0] == self.num_items
        scores = th.matmul(u_embedding, i_embeddings.T)
        return scores.view(-1)

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = th.sum(th.mul(users, pos_items), dim=1)
        neg_scores = th.sum(th.mul(users, neg_items), dim=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        mf_loss = -1 * th.mean(maxi)
        regularizer = (th.norm(users) ** 2
                       + th.norm(pos_items) ** 2
                       + th.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / users.shape[0]
        return mf_loss + emb_loss, mf_loss, emb_loss

    def forward(self , **kwargs):
        pass






