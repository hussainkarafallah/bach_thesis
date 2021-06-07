import dgl
import torch as th
from torch.utils.data import get_worker_info , IterableDataset

def add_graph(train_data):

    train_data.num_users = train_data.dataset.user_num
    train_data.num_items = train_data.dataset.item_num

    _u = th.Tensor(train_data.inter_matrix().row).long()
    _v = th.Tensor(train_data.inter_matrix().col).long()
    _v += train_data.num_users

    u = th.cat((_u , _v) , dim = 0)
    v = th.cat((_v , _u) , dim = 0)


    g = dgl.graph((u , v))
    train_data.graph = g

    znodes = []
    for i , x in enumerate(g.out_degrees().numpy()):
        if x == 0:
            znodes.append(i)

    print("Warning :: there are {} nodes with 0 degrees , nodes {}. Adding self loops".format(len(znodes) , znodes))
    g.add_edges(th.Tensor(znodes).long() , th.Tensor(znodes).long())
    g.ndata['norm'] = th.pow(g.in_degrees().float(), -0.5).reshape(-1 , 1)


    return train_data


class SamplingDataset(IterableDataset):

    def __init__(self, graph, seed_nodes, sampler, sampler_kwargs , postprocessor):
        self.graph = graph
        self.seeds = seed_nodes
        self.fn = sampler
        self.kwargs = sampler_kwargs
        self.post = postprocessor

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is not None
        per_worker = len(self.seeds) // worker_info.num_workers
        worker_id = worker_info.id
        st = worker_id * per_worker
        en = min((worker_id + 1) * per_worker , len(self.seeds))
        yield from self.post( self.fn(self.graph , self.seeds[st:en] , **self.kwargs) )