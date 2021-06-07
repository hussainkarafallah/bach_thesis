from recbole.trainer import Trainer
from torch.utils.data import DataLoader
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
from dgl.dataloading import NodeCollator
import commons
from sortedcontainers import SortedList
import torch as th
from tqdm import tqdm
from recbole.utils import calculate_valid_score



class StochasticTrainer(Trainer):

    def __init__(self, config, model):
        super(StochasticTrainer, self).__init__(config, model)
        self.graph = model.cpu_graph
        self.num_users = self.model.num_users
        self.fans = model.fans
        assert len(self.fans) == model.num_layers

        self.train_loader = None
        self.val_loader = None
        self.sampler = MultiLayerNeighborSampler(self.fans)
        self.collator = NodeCollator(self.graph , self.graph.nodes() , self.sampler)
        self.ITEM_ID = model.ITEM_ID
        self.NEG_ITEM_ID = model.NEG_ITEM_ID


    def collate_fn(self , interaction):
        users_orig = interaction['user_id']
        pos_orig = interaction[self.ITEM_ID] + self.num_users
        try:
            neg_orig = interaction[self.NEG_ITEM_ID] + self.num_users
        except Exception:
            neg_orig = th.zeros((0,))

        unique = SortedList(set(users_orig.numpy().tolist() + pos_orig.numpy().tolist() + neg_orig.numpy().tolist()))
        all_nodes = th.Tensor(list(unique)).long()

        users = th.Tensor([unique.index(x) for x in users_orig]).long()
        pos = th.Tensor([unique.index(x) for x in pos_orig]).long()
        neg = th.Tensor([unique.index(x) for x in neg_orig]).long()

        collations = self.collator.collate(all_nodes)
        _ , _ , blocks = collations
        return users , pos , neg , blocks



    def init_train_loader(self , train_data):
        self.train_loader = DataLoader( list(train_data) , batch_size=None, collate_fn=self.collate_fn,
                                      worker_init_fn=commons.seed_worker, num_workers=commons.workers)


    def _train_epoch(self, train_data, epoch_idx, show_progress=False , **kwargs):
        self.model.train()
        self.init_train_loader(train_data)
        total_loss = 0.
        iterations = 0

        train_iterator = self.train_loader
        if show_progress:
            train_iterator = tqdm(train_iterator)
        for x in train_iterator:
            iterations += 1
            users , pos , neg , blocks = x
            blocks = [x.to(commons.device) for x in blocks]
            self.optimizer.zero_grad()
            loss = self.model.calculate_loss((users , pos , neg , blocks))
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        total_loss /= iterations
        return total_loss

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False ,mode = "validation"):
        self.model.inference(mode)
        return super(StochasticTrainer, self).evaluate(eval_data , load_best_model , model_file , show_progress)


