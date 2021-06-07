import torch.nn as nn
import dgl
import torch as th
import torch.nn.functional as thF
import dgl.function as GF
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from tqdm import tqdm
import commons

initializer = nn.init.xavier_uniform_

class LGCNLayer(nn.Module):

    def __init__(self):
        super(LGCNLayer, self).__init__()



    def reset_parameters(self):
        pass


    @staticmethod
    def msg(edges):
        return {
            'm': edges.src['norm'] * edges.src['x'] * edges.dst['norm']
        }


    def forward_block(self , block , h):
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata['x'] = h_src
            block.dstdata['x'] = h_dst
            block.update_all(self.msg , GF.sum('m' , 'y'))
            return block.dstdata['y']


    def forward_graph(self , graph , h):
        with graph.local_scope():
            graph.ndata['norm'] = graph.ndata['norm']
            graph.ndata['x'] = h
            self.pass_messages(graph)
            ret = graph.ndata['res']
            return ret


class StochasticLGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE
    __name__ = 'S-NGCF'
    default_params = {
        'embedding_size': 64,
        'n_layers' : 3,
        'reg_weight': 1e-5,
        'train_batch_size': 2048,

    }

    def __init__(self , config, dataset):
        super(StochasticLGCN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.num_layers = config['n_layers']
        self.decay = config['reg_weight']


        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(LGCNLayer())
        self.layers = nn.ModuleList(self.layers)
        self.fans = config['fans']
        if not self.fans:
            self.fans = [None] * self.num_layers

        self.cpu_graph = dataset.graph
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.embedding = nn.Embedding(self.cpu_graph.num_nodes() , self.embedding_dim)
        self.check_point = th.zeros( (self.cpu_graph.num_nodes() , self.embedding_dim) , requires_grad=False).to(commons.device)
        self.reset_parameters()


    def reset_parameters(self):
        initializer(self.embedding.weight)
        for x in self.layers:
            x.reset_parameters()

    def forward_blocks(self, blocks , users, pos_items, neg_items):
        assert len(blocks) == len(self.layers) , "{} , {}".format(blocks , self.layers)
        x = self.embedding(blocks[0].srcdata['_ID'])
        all_embeddings = [x]
        for k , (block , lgcn) in enumerate(zip(blocks , self.layers)):
            ret = lgcn.forward_block(block , all_embeddings[-1][:block.number_of_src_nodes()])
            all_embeddings.append(ret)

        ret_users = th.mean(th.stack([ arr[users] for arr in all_embeddings] ,   dim = 0) , dim = 0)
        ret_pos = th.mean(th.stack([ arr[pos_items] for arr in all_embeddings] , dim = 0) , dim = 0)
        ret_neg = th.mean(th.stack([ arr[neg_items] for arr in all_embeddings] , dim = 0) , dim = 0)

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






