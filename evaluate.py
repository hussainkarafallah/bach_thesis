import os , argparse , logging
import utils
import torch
from recbole.utils import DataLoaderType
from recbole.data import data_preparation , create_dataset
from utils.data import add_graph
from recbole.utils import init_seed , init_logger
import commons, statics
from collections import defaultdict , OrderedDict
import pandas
from recbole.evaluator import TopKEvaluator
import json
import numpy as np
global_dict = OrderedDict()
all_metrics = ['recall' , 'ndcg' , 'precision' , 'hit' , 'novelty' , 'diversity']

class CustomEvaluator(TopKEvaluator):
    def __init__(self, proxy , config, metrics):
        super().__init__(config, metrics)
        self.proxy = proxy

    def get_topk(self, batch_matrix_list, eval_data):

        pos_len_list = eval_data.get_pos_len_list()
        batch_result = torch.cat(batch_matrix_list, dim=0).cpu().numpy()

        # unpack top_idx and shape_matrix
        topk_idx = batch_result[:, :-1]
        shapes = batch_result[:, -1]

        assert len(pos_len_list) == len(topk_idx)
        return topk_idx

    def div_nov(self , num_users , num_items , item_degrees , eval_data , adjsets):
        import itertools

        proxy = self.proxy
        if eval_data.dl_type == DataLoaderType.FULL:
            if proxy.item_tensor is None:
                proxy.item_tensor = eval_data.get_item_feature().to(proxy.device).repeat(eval_data.step)
            proxy.tot_item_num = eval_data.dataset.item_num

        batch_matrix_list = []
        iter_data = enumerate(eval_data)

        nov_vals = []
        div_vals = []

        for batch_idx, batched_data in iter_data:
            interaction, scores = proxy._full_sort_batch_eval(batched_data)
            batch_matrix = self.collect(interaction, scores)
            batch_matrix_list.append(batch_matrix)

        topk = self.get_topk(batch_matrix_list, eval_data)
        degs = item_degrees.numpy()

        assert degs.shape[0] == num_items


        for i , subtensor in enumerate(topk):
            cc = []
            for x in subtensor:
                if degs[x] == 0:
                    continue
                val = np.log2( degs[x] / num_users )
                cc.append(val)
            if cc:
                nov_vals.append(- np.mean(cc))

        for i , subtensor in enumerate(topk):
            cc = []
            subtensor = subtensor.tolist()
            for x , y in itertools.combinations(subtensor , 2):
                if len(adjsets[x].union(adjsets[y])) == 0:
                    cc.append(0)
                else:
                    cc.append( -np.log2( 1 + len(adjsets[x].intersection(adjsets[y])) / len(adjsets[x].union(adjsets[y]))  ) )
            if cc:
                div_vals.append(np.mean(cc))

        return np.mean(nov_vals) ,  np.mean(div_vals)


def construct_sets(train_data):
    spmat = train_data.graph.adjacency_matrix(transpose=True, scipy_fmt='coo')
    adjsets = defaultdict(set)
    for x, y in zip(spmat.row, spmat.col):
        if x < train_data.num_users and y >= train_data.num_users:
            adjsets[y - train_data.num_users].add(x)
    return adjsets

def run_evaluation(model_name , dataset_name , model_path):

    global_dict[model_name] = OrderedDict()
    for metric in all_metrics:
        global_dict[model_name][metric] = OrderedDict()

    kvals = [10,20,30]

    dataset_initialized = False
    train_data = None
    adj_sets = None

    for K in kvals:

        commons.init_seeds()

        model_class = statics.model_name_map[model_name]
        model_path = os.path.join("bestmodels" , dataset_name , str(K) , "{}.pth".format(model_name))
        loaded_file = torch.load(model_path)
        config = loaded_file['config']
        config['data_path'] = os.path.join('dataset' , dataset_name)
        config['topk'] = K
        config['valid_metric'] = 'Recall@{}'.format(K)
        config['eval_batch_size'] = 500000
        init_seed(config['seed'], config['reproducibility'])


        init_logger(config)
        logger = logging.getLogger()


        if not dataset_initialized:
            # dataset filtering
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            train_data = add_graph(train_data)
            item_degrees = train_data.graph.in_degrees()[train_data.num_users : ]
            adj_sets = construct_sets(train_data)
            dataset_initialized = True

        assert adj_sets
        assert train_data

        model = model_class(config, train_data).to(commons.device)
        trainer = utils.get_trainer(config)(config, model)

        test_result = trainer.evaluate(test_data , load_best_model=True , model_file=model_path)
        custom_evaluator = CustomEvaluator(trainer , config , config['metrics'])
        novelty , diversity = custom_evaluator.div_nov(train_data.num_users , train_data.num_items , item_degrees , test_data , adj_sets)
        novelty = round(novelty , 4)
        diversity = round(diversity , 4)
        for metric in all_metrics:
            if metric not in ['novelty' , 'diversity']:
                global_dict[model_name][metric][K] = test_result["{}@{}".format(metric , K)]
        global_dict[model_name]['novelty'][K] = novelty
        global_dict[model_name]['diversity'][K] = diversity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, action='store', help="dataset name" , required=True)
    parser.add_argument("--models_path" , type = str , action='store' , help = "models_path")
    parser.add_argument("--out" , type = str , action='store' , help = "output file")
    args, unknown = parser.parse_known_args()

    dataset_name = args.dataset
    mpath = args.models_path

    all_models = ['ItemKNN' , 'BPR',  'NeuMF' , 'SpectralCF' , 'GCMC' , 'NGCF' , 'LightGCN']
    for model in all_models:
        run_evaluation(model , dataset_name , model_path = mpath)

    if args.out:
        with open(args.out , 'w') as f:
            json.dump(global_dict , f, indent = 2)
    else:
        print(json.dumps(global_dict, indent=2))

    # ['ItemKNN']: #