import random
import numpy as np
import torch
import os

seed = 2020
device = 'cuda'

def init_seeds():
    global seed , device
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


workers = min(os.cpu_count() - 2 , 12)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

root_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(root_dir , "cache")
os.makedirs(cache_dir , exist_ok=True)

tuning_results_dir = os.path.join(root_dir , 'tuning_results')
os.makedirs(tuning_results_dir , exist_ok=True)

head_log_dir = os.path.join(root_dir , "logs")
os.makedirs(head_log_dir, exist_ok=True)
os.makedirs(os.path.join(root_dir , 'data/walks') , exist_ok=True)

#log_dir = os.path.join(head_log_dir , datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#global_writer = SummaryWriter(log_dir)


logger_name = 'recbole'

print("Initialized All seeds")
