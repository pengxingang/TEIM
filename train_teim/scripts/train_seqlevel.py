import sys
sys.path.append('.')
import numpy as np
import pandas as pd
import shutil
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
pl.seed_everything(0)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from scripts.models import TEIM
from utils.misc import load_config, calc_auc_aupr
from utils.dataset import load_data
import os


class SeqLevelSystem(pl.LightningModule):
    def __init__(self, config, train_set, val_set):
        super().__init__()
        self.config = config
        self.lr = config.training.lr
        self.teim_seq = TEIM(config.model)
        self.train_set = train_set
        self.val_set = val_set
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.training.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config.training.batch_size, shuffle=False)

    def forward(self, x):
        return self.teim_seq(x)['seqlevel_out']
    
    def minimum_step(self, batch, device=None):
        # batch = batch.to(self.device)
        if device is None:
            cdr3, epi, labels = batch['cdr3'], batch['epi'], batch['labels']
        else:
            cdr3, epi, labels = batch['cdr3'].to(device), batch['epi'].to(device), batch['labels'].to(device)
        pred = self([cdr3, epi])
        loss = self.get_loss(pred, labels)
        return loss, labels, pred

    def training_step(self, batch, batch_idx):
        self.train()
        loss, labels, pred = self.minimum_step(batch)
        self.log('train/loss', loss)
        return {
            'loss':loss,
            'labels': labels,
            'pred': pred
        }

    def training_epoch_end(self, training_step_outputs):
        
        ## training metric
        loss, auc, aupr, auc_mean, aupr_mean = self.evaluate_model(self.train_dataloader())

        print('Train set: AUC={:.4}, AUPR={:.4}, AUC_AVG={:.4}, AUPR_AVG={:.4}'.format(auc, aupr, auc_mean, aupr_mean))
        self.log('lr', self.optimizers().state_dict()['param_groups'][0]['lr'])
        self.log_dict({
            'train/auc':auc,
            'train/aupr':aupr,
            'train/auc_avg':auc_mean,
            'train/aupr_avg':aupr_mean,
        }, prog_bar=False)

        

        ## validating metric
        loss, auc, aupr, auc_mean, aupr_mean = self.evaluate_model(self.val_dataloader())
        print('Valid', ' set: AUC={:.4}, AUPR={:.4}, AUC_AVG={:.4}, AUPR_AVG={:.4}'.format(auc, aupr, auc_mean, aupr_mean))
        self.log_dict({
            'valid/loss':loss,
            'valid/auc':auc,
            'valid/aupr':aupr,
            'valid/auc_avg':auc_mean,
            'valid/aupr_avg':aupr_mean,
        }, prog_bar=False)


    def evaluate_model(self, data_loader=None, ):
        self.eval()
        loss = 0
        y_true, y_pred = [], []
        epi_ids = []

        for i, batch in enumerate(data_loader):
            loss_this, y, y_hat = self.minimum_step(batch, self.device)
            loss += loss_this.item()
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(y_hat.detach().cpu().numpy().tolist())
            if 'epi_id' in batch:
                epi_ids.extend(batch['epi_id'].cpu().numpy().tolist())
        loss /= (i+1)
        auc, aupr = self.get_scores(y_true, y_pred)
        ## per epi auc
        if len(epi_ids) > 0:
            ids_uni = np.unique(epi_ids, axis=0)

            auc_sum = 0
            aupr_sum = 0
            cnt = 0
            for i, id_ in enumerate(ids_uni):
                index = np.array(epi_ids == id_)
                y_true_epi = np.array(y_true)[index]
                y_pred_epi = np.array(y_pred)[index]
                auc_epi, aupr_epi = self.get_scores(y_true_epi, y_pred_epi)
                if auc_epi is None:
                    continue
                auc_sum += auc_epi
                aupr_sum += aupr_epi
                cnt += 1
            auc_mean = auc_sum / cnt
            aupr_mean = aupr_sum / cnt
        else:
            auc_mean, aupr_mean = auc, aupr

        return loss, auc, aupr, auc_mean, aupr_mean


    def predict(self, data_loader=None):
        self.eval()
        cdr3_seqs, epi_seqs, y_true, y_pred = [], [], [], []
        epi_ids = []

        for i, batch in tqdm(enumerate(data_loader), desc='Predicting'):
            loss, y, y_hat = self.minimum_step(batch, self.device)
            cdr3_seqs.extend(batch['cdr3_seqs'])
            epi_seqs.extend(batch['epi_seqs'])
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(y_hat.detach().cpu().numpy().tolist())
            if 'epi_id' in batch.keys():
                epi_ids.extend(batch['epi_id'].cpu().numpy().tolist())

        if len(epi_ids) > 0:
            return cdr3_seqs, epi_seqs, y_true, np.reshape(y_pred, -1), epi_ids
        else:
            return cdr3_seqs, epi_seqs, y_true, np.reshape(y_pred, -1)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_loss(self, pred, labels):
        loss = F.binary_cross_entropy(pred.view(-1), labels.float(), weight=None, reduction='mean')
        return loss

    def get_scores(self, y_true, y_pred):
        if len(np.unique(y_true)) == 1:
            return None, None
        else:
            return calc_auc_aupr(y_true, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/seqlevel_cv_shuffle.yml')
    parser.add_argument('--config', type=str, default='configs/seqlevel_all.yml')
    args = parser.parse_args()
    # load config
    config_path = args.config
    
    # load config
    config = load_config(config_path)

    # load data
    print('Loading data...')
    datasets = load_data(config.data)
    train_set, val_set = datasets['train'], datasets['val']
    
    for i_split, (train_set_this, val_set_this) in enumerate(zip(train_set, val_set)):
        print('Split {}'.format(i_split), 'Train:', len(train_set_this), 'Val:', len(val_set_this))
        # load model and trainer
        print('Loading model and trainer...')
        model = SeqLevelSystem(config, train_set_this, val_set_this)
        checkpoint = ModelCheckpoint(monitor='valid/auc_avg', save_last=True, mode='max', save_top_k=1)
        earlystop = EarlyStopping(monitor='valid/auc_avg', patience=15, mode='max')
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            gpus=1,
            callbacks=[checkpoint, earlystop],
            default_root_dir=os.path.join(os.getcwd(), 'logs', config.name)
        )
        print('Num of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

        # train
        print('Training...')
        trainer.fit(model, )
        shutil.copy2(config_path, os.path.join(trainer.log_dir, os.path.basename(config_path)))
        
        # predict val
        print('Predicting val...')
        results = model.predict(model.val_dataloader())
        print('Saving results...')
        if len(results) == 4:
            pd.DataFrame(zip(*results), columns=['cdr3', 'epi', 'y_true', 'y_pred']).to_csv(os.path.join(trainer.log_dir, 'val_pred.csv'), index=False)
        else:
            pd.DataFrame(zip(*results), columns=['cdr3', 'epi', 'y_true', 'y_pred', 'epi_id']).to_csv(os.path.join(trainer.log_dir, 'val_pred.csv'), index=False)
        print('Done')