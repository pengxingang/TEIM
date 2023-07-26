import sys
sys.path.append('.')
import shutil
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import StepLR

from scripts.models import TEIM
from utils.misc import get_scores_dist, get_scores_contact, load_model_from_ckpt, load_config
from utils.encoding import decoding_one_mat
from utils.dataset import load_data

pl.seed_everything(0)


class ResLevelSystem(pl.LightningModule):
    def __init__(self, config, train_set, val_set):
        super().__init__()
        self.config = config

        self.teim_res = TEIM(config.model, )
        self.from_pretrained(config.pretraining)
        self.train_set = train_set
        self.val_set = val_set
        self.lr = config.training.lr
        self.patience=config.training.patience
        self.decay = config.training.decay
    
    def from_pretrained(self, pretraining):
        print('Loading pretrained model: {}'.format(pretraining.path))
        self.teim_res = load_model_from_ckpt(pretraining.path, self.teim_res)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.training.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config.training.batch_size, shuffle=False)

    def forward(self, x, *args, **kwargs):
        return self.teim_res(x, *args, **kwargs)['reslevel_out']
    
    def minimum_step(self, batch, device=None):
        if device is not None:
            batch.update({k:v.to(device) for k,v in batch.items() if type(v) is torch.Tensor})
        
        cdr3, epi, mask = batch['cdr3'], batch['epi'], batch['mask_mat']
        dist, contact = batch['dist_mat'], batch['contact_mat']
        addition = {}
        if 'mhc' in batch.keys():
            mhc = batch['mhc']
            addition['mhc'] = mhc
        pred = self([cdr3, epi], addition)
        loss = self.get_loss(pred, [dist, contact, mask])
        return loss, pred, dist, contact, mask

    def training_step(self, batch, batch_idx):
        self.train()
        loss_dict, pred, dist, contact, mask = self.minimum_step(batch)
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss_dict, pred, dist, contact, mask = self.minimum_step(batch)

        for key, value in loss_dict.items():
            self.log(f'val/{key}', value)
        return loss_dict['loss']

    def training_epoch_end(self, training_step_outputs):
        
        ## validating metric
        loss, scores, scores_samples = self.evaluate_model(self.val_dataloader())
        self.log_dict({
            'val/corr': scores[0],
            'val/mse': scores[1],
            'val/mape': scores[2],
            'val/auc': scores[3],
        })

    def evaluate_model(self, data_loader=None, ):
        self.eval()
        loss_dict = {'loss': 0, 'loss_dist': 0, 'loss_contact': 0}
        pred, dist, contact, mask = [], [], [], []

        for i, batch in enumerate(data_loader):
            loss_this, pred_, dist_, contact_, mask_ = self.minimum_step(batch, self.device)
            for key, value in loss_this.items():
                loss_dict[key] += value
            pred.extend(pred_.detach().cpu().numpy().tolist())
            dist.extend(dist_.detach().cpu().numpy().tolist())
            contact.extend(contact_.detach().cpu().numpy().tolist())
            mask.extend(mask_.detach().cpu().numpy().tolist())
        for key, value in loss_dict.items():
            loss_dict[key] /= len(data_loader)

        scores, scores_samples = self.get_scores(pred, [dist, contact, mask])
        return loss_dict, scores, scores_samples
        

    def predict(self, data_loader=None):
        self.eval()
        pdb_chains = []
        for i, batch in enumerate(data_loader):
            pdb_chains.extend(batch['pdb_chains'])
        return [pdb_chains]

    def predict_valset(self):
        self.eval()
        pred = {}
        cdr3_list = []
        epi_list = []
        for batch in self.val_dataloader():
            pdb_chain = batch['pdb_chains']
            cdr3 = batch['cdr3_seqs']
            epi = batch['epi_seqs']
            loss_, pred_, dist_, contact_, mask_ = self.minimum_step(batch, self.device)
            for i, pdb in enumerate(pdb_chain):
                value = pred_.detach().cpu().numpy()[i]
                pred[pdb] = decoding_one_mat(value, len(cdr3[i]), len(epi[i]))
                cdr3_list.append(cdr3[i])
                epi_list.append(epi[i])
            
        return pred, cdr3_list, epi_list


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': StepLR(optimizer, self.patience, gamma=self.decay)
            }

    def get_loss(self, pred, labels):
        dist, contact, mask = labels
        loss_dist = F.mse_loss(pred[..., 0], dist, reduction='none')
        loss_dist = loss_dist * mask
        loss_dist = torch.sum(loss_dist) / torch.sum(mask)

        loss_bd = F.binary_cross_entropy(pred[..., 1], contact.float(), reduction='none')
        loss_bd = loss_bd * mask
        loss_bd = torch.sum(loss_bd) / torch.sum(mask)

        loss = loss_dist + 1 * loss_bd
        return {
            'loss': loss,
            'loss_dist': loss_dist,
            'loss_contact': loss_bd
        }

    def get_scores(self, pred, labels):
        dist, contact, mask = labels
        avg_metrics_dist, metrics_dist = get_scores_dist(np.array(dist), np.array(pred)[..., 0], np.array(mask))
        avg_metrics_bd, metrics_bd = get_scores_contact(np.array(contact), np.array(pred)[..., 1], np.array(mask))
        return avg_metrics_dist + avg_metrics_bd, metrics_dist + metrics_bd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/reslevel_bothnew.yml')
    args = parser.parse_args()
    # load config
    config_path = args.config
    config = load_config(config_path)

    # load data
    print('Loading data...')
    datasets = load_data(config.data)
    train_set, val_set = datasets['train'], datasets['val']
    
    for i_split, (train_set_this, val_set_this) in enumerate(zip(train_set, val_set)):
        print('Split {}'.format(i_split), 'Train:', len(train_set_this), 'Val:', len(val_set_this))
        # load model and trainer
        print('Loading model and trainer...')
        model = ResLevelSystem(config, train_set_this, val_set_this)
        checkpoint = ModelCheckpoint(monitor='val/loss', save_last=True, mode='min', save_top_k=1)
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            gpus=1,
            callbacks=[checkpoint],
            default_root_dir=os.path.join(os.getcwd(), 'logs', config.name)
        )
        print('Num of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

        # train
        print('Training...')
        trainer.fit(model, )
        shutil.copy2(config_path, os.path.join(trainer.log_dir, os.path.basename(config_path)))
        
        # predict val
        print('Predicting val...')
        pred, cdr3_list, epi_list = model.predict_valset()

        # save results
        save_path = os.path.join(trainer.log_dir, 'val_pred')
        os.makedirs(save_path, exist_ok=True)
        print('Saving results...')
        for (key, value), cdr3, epi in zip(pred.items(), cdr3_list, epi_list):
            df = pd.DataFrame(value[..., 0], index=list(cdr3), columns=list(epi))
            df.to_csv(os.path.join(save_path, 'dist_' + key+'.csv'))
            df = pd.DataFrame(value[..., 1], index=list(cdr3), columns=list(epi))
            df.to_csv(os.path.join(save_path, 'contact_' + key+'.csv'))

        print('Done')