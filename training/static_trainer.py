from recbole.trainer import Trainer
class StaticTrainer(Trainer):

    def __init__(self, config, model):
        super(StaticTrainer, self).__init__(config, model)
        assert config['epochs'] == 1 , "Trainer for static models should have only epoch"

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        return 0





