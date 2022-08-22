
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('.')

import torch
from torch import nn
from copy import deepcopy
import pytorch_lightning as pl
from tqdm.std import tqdm
from data_process import Tokenizer, PretrainedEncoder, encode_epi, encode_cdr3, load_ae_model
from models import NewCNN2dBaseline
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from copy import deepcopy
from scipy.stats import pearsonr
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list[0])
    def __getitem__(self, idx):
        return [data[idx, ...] for data in self.data_list]


class ContactCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding_module = model.embedding_module
        self.seq_cdr3 = model.seq_cdr3
        self.seq_epi = model.seq_epi
        self.cnn_list = model.cnn_list
        self.global_linear = model.global_linear
        self.cnn_out = nn.Sequential(
                        nn.Conv2d(
                            in_channels=model.cnn_list[-2][0].cnn.out_channels,
                            out_channels=2,
                            kernel_size=5,
                            padding=2
                        ),
                        # nn.ReLU(),  # * bd prediction
        )

    def forward(self, inputs):
        cdr3, epi, pretrained = inputs
        cdr3_aa, epi_aa = cdr3.long(), epi.long()
        pretrained = pretrained.float()
        len_cdr3, len_epi = cdr3_aa.shape[1], epi_aa.shape[1]
        # embedding
        cdr3_emb = self.embedding_module(cdr3_aa)
        epi_emb = self.embedding_module(epi_aa)
        # sequence features
        cdr3_feat = self.seq_cdr3(cdr3_emb.transpose(1, 2))
        epi_feat = self.seq_epi(epi_emb.transpose(1, 2))
        # inter module
        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])
        epi_feat_mat = epi_feat.unsqueeze(2).repeat([1, 1, len_cdr3, 1])
        inter_map = cdr3_feat_mat * epi_feat_mat

        vec = pretrained.unsqueeze(2).unsqueeze(3)
        n_layers = len(self.cnn_list)
        for i in range(n_layers - 1):
            if i == n_layers - 2:  # the last cnn layer
                global_feat = self.global_linear(vec)
                inter_map = self.cnn_list[i][0](inter_map)
                inter_map = inter_map + global_feat
                inter_map = self.cnn_list[i][1](inter_map)
            else:
                inter_map = self.cnn_list[i](inter_map)

        # output module
        cnn_out = self.cnn_out(inter_map)
        out_dist = torch.relu(cnn_out[:, 0, :, :])
        out_bd = torch.sigmoid(cnn_out[:, 1, :, :])  # * bd prediction
        cnn_out = torch.cat([out_dist.unsqueeze(-1), out_bd.unsqueeze(-1)], axis=-1)
        return cnn_out


def build_model(base_model_path='./ckpt/base_model.ckpt'):

    # # load base model (TEIM-Res)
    base_model = TrainerSystem.load_from_checkpoint(base_model_path)
    # data_mode = base_model.hparams['data_mode']
    model_args = base_model.hparams['model_args']
    tokenizer = model_args['tokenizer']

    # # new model (TEIM-Samp)
    model = ContactCNN(base_model.model)

    print('Model args: ', model_args)
    return model, tokenizer


class TrainerSystem(pl.LightningModule):
    def __init__(self, data_mode, model_args, lr, weight_decay, model=None, setting=''):
        super().__init__()
        self.data_mode = data_mode
        self.save_hyperparameters('data_mode', 'model_args', 'lr', 'weight_decay')
        if model is None:
            self.model = NewCNN2dBaseline(model_types=data_mode['model_type'].split('|'), **model_args)
        else:
            self.model = model
        self.losses = []
        self.metrics = []
        self.metrics_best = [[0 for _ in range(4)] for _ in range(2)]

        # pretraining ablation
        if setting == 'random':   # no pretraining
            self.random_weights()
        self.last = False
        if 'last' in setting:  # finetune last
            self.last = True
            self.frozen_layers()

        self.scores_epochs = []

    def random_weights(self):
        for m in self.model.modules():
            classname = m.__class__.__name__
            if (classname.find('Conv') != -1):
                nn.init.normal(m.weight.data)
                nn.init.uniform_(m.bias.data)
            elif (classname.find('Linear') != -1):
                # nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.weight.data)
                nn.init.uniform_(m.bias.data)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data)
                nn.init.uniform_(m.bias.data)

    def frozen_layers(self):
        # pass
        for params in self.model.parameters():
            params.requires_grad = False
        for params in self.model.cnn_out.parameters():
            params.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.train()
        loss, y, y_hat, mask = self.minimum_step(batch)
        self.log('train_loss', loss)
        return {
            'loss': loss,
            'y_hat': y_hat,
            'y': y
        }

    def training_epoch_end(self, training_step_outputs):
        loss, scores, _, _, _ = self.evaluate_model(self.val_dataloader())
        self.log('val_loss', loss)
        self.log_dict({
            'val_corr': scores[0],
            'val_mse': scores[1],
            'val_mape': scores[2],
            'val_auc': scores[3],
        })
        self.scores_epochs.append(scores)
        # self.mse_epochs.append(mse)

    def minimum_step(self, batch):
        x, y = batch[0:3], batch[3].float().to(self.device)
        mask = batch[4].float().to(self.device)
        y_hat = self([x_.to(self.device) for x_ in x])

        y_bind_hat = y_hat
        y_bind_true = y

        loss = self.get_loss(y_bind_hat, y_bind_true, mask)
        return loss, y_bind_true, y_bind_hat, mask

    def evaluate_model(self, data_loader=None, plot_model=False, device=None):
        if device is not None:
            self.to(device)
        else:
            device = self.device
        self.eval()

        y_true = []
        y_pred = []
        y_mask = []
        for i, batch in enumerate(data_loader):
            # x.extend(batch[1].detach().cpu().numpy())
            loss, y, y_hat, mask = self.minimum_step(batch)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.detach().cpu().numpy())
            y_mask.extend(mask.detach().cpu().numpy())

        scores, scores_samples = self.get_scores(y_true, y_pred, y_mask)
        return loss, scores, scores_samples, y_true, y_pred

    def configure_optimizers(self):
        if self.last:
            # last_params = self.model.dense_module.parameters()
            # last_layer_params_id = list(map(id, last_params))
            # base_params = filter(lambda p:id(p) not in last_layer_params_id, self.model.parameters())
            # optimizer = torch.optim.AdamW([{'params':last_params}, {'params':base_params, 'lr':1e-4}],  lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {
                'optimizer': optimizer,
                'lr_scheduler': StepLR(optimizer, 200, gamma=0.5, last_epoch=-1, ),
                # 'lr_scheduler': ReduceLROnPlateau(optimizer, factor=0.2, patience=20, min_lr=1e-5),
                # 'monitor': 'valid0_loss'
                # 'monitor': 'valid_loss'
                }
        # return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def get_loss(self, y_hat, y, mask):
        # loss = F.binary_cross_entropy(y_hat, y)
        # l1 = 6e-3
        loss_dist = F.mse_loss(y_hat[..., 0], y[..., 0], reduction='none')
        loss_dist = loss_dist * mask
        loss_dist = torch.mean(torch.mean(loss_dist, dim=-1), dim=-1)
        loss_dist = torch.sum(loss_dist)

        loss_bd = F.binary_cross_entropy(y_hat[..., 1], y[..., 1], reduction='none')  # * bd site
        loss_bd = loss_bd * mask
        loss_bd = torch.mean(torch.mean(loss_bd, dim=-1), dim=-1)
        loss_bd = torch.sum(loss_bd)

        loss = loss_dist + 1 * loss_bd
        return loss

    def get_scores(self, y_true, y_pred, y_mask):
        avg_metrics_dist, metrics_dist = self.get_scores_dist(np.array(y_true)[..., 0], np.array(y_pred)[..., 0], y_mask)
        avg_metrics_bd, metrics_bd = self.get_scores_bd(np.array(y_true)[..., 1], np.array(y_pred)[..., 1], y_mask)
        return avg_metrics_dist + avg_metrics_bd, metrics_dist + metrics_bd

    def get_scores_bd(self, y_true, y_pred, y_mask):
        coef = []
        for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):
            y_true_, y_pred_, y_mask_ = y_true_.reshape([-1]), y_pred_.reshape([-1]), y_mask_.reshape([-1]).astype('bool')
            y_true_ = y_true_[y_mask_]
            y_pred_ = y_pred_[y_mask_]
            try:
                coef_ = roc_auc_score(y_true_, y_pred_)  # * binding site
            except Exception:
                coef_ = 0
            coef.append(coef_)

        avg_coef = np.mean(coef)
        return [avg_coef], [coef]

    def get_scores_dist(self, y_true, y_pred, y_mask):
        coef = []
        mae = []
        mape = []
        for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):
            y_true_, y_pred_, y_mask_ = y_true_.reshape([-1]), y_pred_.reshape([-1]), y_mask_.reshape([-1]).astype('bool')
            y_true_ = y_true_[y_mask_]
            y_pred_ = y_pred_[y_mask_]
            try:
                coef_, _ = pearsonr(y_true_, y_pred_)
            except Exception:
                coef_ = 0
            coef.append(coef_)

            mae_ = median_absolute_error(y_true_, y_pred_)
            mae.append(mae_)

            mape_ = np.median(np.abs((np.array(y_true_) - np.array(y_pred_)) / np.array(y_true_)))
            mape.append(mape_)
        avg_coef = np.mean(coef)
        avg_mae = np.mean(mae)
        avg_mape = np.mean(mape)
        return [avg_coef, avg_mae, avg_mape], [coef, mae, mape]


class PdbPredictor:
    def __init__(self, ):
        self.data_mode = {
            'form': 'numbering|center+epiid+ae',
            'model_type': '2dcnn_ae_new',
            'threshold': 0.5,
            'use_vae': False,
        }
        self.model, self.tokenizer = self.load_pdb_model()
        self.model = self.model.to(device)
        self.model.eval()

    def load_pdb_model(self):
        pdb_model_path = './ckpt/pdb_model.ckpt'
        base_model, tokenizer = build_model()
        model = TrainerSystem(self.data_mode, dict(
                tokenizer=Tokenizer(),
                dim_hidden = 256,
                form = self.data_mode['form'],
                model_type = self.data_mode['model_type']
            ), lr=0.1, weight_decay=0.1, model=deepcopy(base_model))

        model.load_state_dict(torch.load(pdb_model_path, map_location=device)['state_dict'])

        n_para = np.sum([a.numel() for a in model.parameters()])
        print('Number of paramters', n_para)
        return model, tokenizer

    def predict(self, seqs_cdr3, seqs_epi):
        encoding_cdr3, encoding_epi, epi_vec, mask = self.encode_data(seqs_cdr3, seqs_epi)
        out = self.forward(encoding_cdr3, encoding_epi, epi_vec)
        predictions = self.make_mask(out, mask)  # delete the padding positions
        return predictions

    def encode_data(self, seqs_cdr3, seqs_epi):

        encoding_cdr3 = encode_cdr3(seqs_cdr3, self.tokenizer)
        epi_encoder = PretrainedEncoder(self.tokenizer)
        encoding_epi, epi_vec = epi_encoder.encode_pretrained_epi(seqs_epi)
        # encoding_epi = encode_epi(seqs_epi)

        # # build mask of the padding position
        n_samples = len(encoding_epi)
        len_cdr3 = len(encoding_cdr3[0])
        len_epi = len(encoding_epi[0])
        encoding_mask = np.zeros([n_samples, len_cdr3, len_epi])
        for idx_sample, (enc_cdr3_this, enc_epi_this) in enumerate(zip(encoding_cdr3, encoding_epi)):
            mask = np.ones([len_cdr3, len_epi])
            zero_cdr3 = (enc_cdr3_this == 0)
            mask[zero_cdr3, :] = 0
            zero_epi = (enc_epi_this == 0)
            mask[:, zero_epi] = 0
            encoding_mask[idx_sample] = mask
        return [encoding_cdr3, encoding_epi, epi_vec, encoding_mask]

    def forward(self, encoding_cdr3, encoding_epi, epi_vec):
        bs = 128
        pred = []
        for batch in np.arange(0, len(encoding_cdr3) // bs + 1):
            idx_end = min((batch+1)*bs, len(encoding_cdr3))
            input_cdr3 = torch.Tensor(encoding_cdr3[batch*bs:idx_end, ...]).to(device)
            input_epi = torch.Tensor(encoding_epi[batch*bs:idx_end, ...]).to(device)
            input_vec = torch.Tensor(epi_vec[batch*bs:idx_end, ...]).to(device)
            pred_batch = self.model([input_cdr3, input_epi, input_vec])
            pred_batch = pred_batch.cpu().detach().numpy()
            pred.extend(pred_batch)
        return np.array(pred)

    def make_mask(self, out, mask):
        out_list = []
        for i in range(len(out)):
            out_, mask_ = out[i], mask[i]
            idx_cdr3 = np.sum(mask_, axis=1) != 0
            out_ = out_[idx_cdr3, :, :]
            idx_epi = np.sum(mask_, axis=0) != 0
            out_ = out_[:, idx_epi, :]
            out_list.append(out_)
        return out_list


def get_seqs(input_path):
    df = pd.read_csv(input_path)
    seqs_cdr3 = df['cdr3'].values
    seqs_epi = df['epitope'].values
    return seqs_cdr3, seqs_epi


def predict_dist_site(input_path, save_dir='./outputs', batch_size=128):
    # # load data
    seqs_cdr3, seqs_epi = get_seqs(input_path)

    # # build predictor
    predictor = PdbPredictor()
    for i in range(0, len(seqs_epi), batch_size):
        print('Predict batch', i // batch_size)
        # # get batch data
        idx_end = min(i+batch_size, len(seqs_epi))
        seqs_cdr3_batch = seqs_cdr3[i:idx_end]
        seqs_epi_batch = seqs_epi[i:idx_end]

        # # predict distance and site
        pred = predictor.predict(seqs_cdr3_batch, seqs_epi_batch)

        # # save
        if not os.path.exists(save_dir):
            os.makedirs(save_dir,)
        for cdr3, epitope, dist_site in tqdm(zip(seqs_cdr3_batch, seqs_epi_batch, pred), desc='Saving prediction...'):
            name_func = lambda x: x + '_' + cdr3 + '_' + epitope + '.csv'
            dist = dist_site[..., 0]
            site = dist_site[..., 1]
            pd.DataFrame(dist, index=list(cdr3), columns=list(epitope)).to_csv(os.path.join(save_dir, name_func('dist')))
            pd.DataFrame(site, index=list(cdr3), columns=list(epitope)).to_csv(os.path.join(save_dir, name_func('site')))
    print('Done. The predictions are in', save_dir)


if __name__ == '__main__':
    input_path = './inputs/inputs.csv'
    output_dir = './outputs'
    predict_dist_site(input_path, output_dir)
    # eval()
