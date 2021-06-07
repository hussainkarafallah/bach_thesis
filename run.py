from models import *
import commons , statics
import argparse , json , os , pandas , logging
from recbole.config import Config
from recbole.data import data_preparation , create_dataset
from utils.data import add_graph
from recbole.utils import init_seed , init_logger

import utils
import torch
from recbole.utils.utils import set_color
import gc
import shutil

def run_trial(model_name , dataset_name , hp_config = None , save_flag = False):

    if not hp_config:
        hp_config = {}
        tuning = False
    else:
        tuning = True

    commons.init_seeds()
    verbose = True
    verbose = (not tuning)
    model_class = statics.model_name_map[model_name]
    try:
        default_config = model_class.default_params
    except AttributeError:
        default_config = {}
        assert model_name in statics.recbole_models

    default_config.update(statics.datasets_params[dataset_name])
    default_config.update(hp_config)

    config = Config(model=model_class, dataset=dataset_name, config_dict=default_config)
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = logging.getLogger()

    # logger initialization
    if verbose:
        logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    train_data = add_graph(train_data)

    if verbose:
        logger.info(dataset)

    model = model_class(config, train_data).to(commons.device)
    trainer = utils.get_trainer(config)(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data , verbose= verbose , show_progress=verbose)
    test_result = trainer.evaluate(test_data)

    if verbose:
        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

    metric = str.lower(config['valid_metric'])

    if save_flag:
        os.makedirs(os.path.join("bestmodels" , dataset_name , str(config["topk"])) , exist_ok=True)
        save_path = os.path.join("bestmodels" , dataset_name , str(config["topk"]) , "{}.pth".format(model_name))
    else:
        save_path = None

    if save_path:
        shutil.copyfile(trainer.saved_model_file , save_path)

    return {
        'metric' : config['valid_metric'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_score': test_result[metric]
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, action='store', help="model name")
    parser.add_argument("--dataset", type=str, action='store', help="dataset name")
    parser.add_argument("--save" , action = 'store_true' , help = "saved model path" , default=False)
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    save_flag= args.save

    run_trial(model_name , dataset_name , save_flag = save_flag)