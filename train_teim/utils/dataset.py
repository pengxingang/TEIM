from multiprocessing import Pool
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

from sklearn.model_selection import GroupKFold
from torch.utils.data import Subset
from utils.encoding import *

DATA_ROOT = '../data'

def load_data(config):
    dataset_name = config.dataset
    split_type = getattr(config, 'split', '')
    if dataset_name == 'seqlevel_data':
        dataset = SeqLevelDataset(config)
        if split_type == 'cv-new_epitope':
            epi2cluster = get_cluster(dataset_name, split_type)
            epi_id = dataset.get_all_epiid()
            group_id = [epi2cluster[epi_this] for epi_this in epi_id]

            kfold = GroupKFold(5)
            splits = kfold.split(epi_id, groups=group_id)
            datasets_cv = {'train':[],'val':[]}
            for train_idx, val_idx in splits:
                datasets_cv['train'].append(Subset(dataset, train_idx))
                datasets_cv['val'].append(Subset(dataset, val_idx))
            return datasets_cv

    elif dataset_name == 'reslevel_data':
        dataset = ResLevelDataset(config)
        if split_type in ['cv-both_new', 'cv-new_cdr3', 'cv-new_epi']:
            cdr32cluster, epi2cluster = get_cluster(dataset_name, split_type)

            kfold = GroupKFold(3) if split_type == 'cv-both_new' else GroupKFold(5)
            if len(cdr32cluster) != 0:  # new_cdr3, both_new
                all_cdr3= dataset.get_all_values('cdr3_seqs')
                cdr3_group_id = [cdr32cluster[cdr3_this] for cdr3_this in all_cdr3]
                split_cdr3 = list(kfold.split(all_cdr3, groups=cdr3_group_id))
                split = split_cdr3
            if len(epi2cluster) != 0:  # new_epi, both_new
                all_epi = dataset.get_all_values('epi_seqs')
                epi_group_id = [epi2cluster[epi_this] for epi_this in all_epi]
                split_epi = list(kfold.split(all_epi, groups=epi_group_id))
                split = split_epi
            if split_type == 'cv-both_new':
                split = [[np.intersect1d(fold_tcr[0], fold_epi[0]), np.intersect1d(fold_tcr[1], fold_epi[1])]
                    for fold_tcr, fold_epi in zip(split_cdr3, split_epi)]
            
            datasets_cv = {'train':[],'val':[]}
            for train_idx, val_idx in split:
                datasets_cv['val'].append(Subset(dataset, val_idx))
                datasets_cv['train'].append(Subset(dataset, train_idx))
            return datasets_cv

    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_name))

    if (split_type is None) or (split_type == ''):
        print('No split specified, using train as default')
        return dataset
    elif split_type == 'train-val':
        train_ratio = getattr(config, 'train_ratio', 0.8)
        index = np.random.permutation(len(dataset))
        train_set = Subset(dataset, index[:int(len(dataset) * train_ratio)])
        val_set = Subset(dataset, index[int(len(dataset) * train_ratio):])
        return {'train': [train_set], 'val': [val_set]}
    else:
        raise ValueError('Unknown split: {}'.format(split_type))


def get_cluster(dataset_name, split):
    if dataset_name == 'seqlevel_data' and split == 'cv-new_epitope':
        cluster_path = os.path.join(DATA_ROOT, 'cluster/seqlevel_epi_cluster_0.5.pkl')
        with open(cluster_path, 'rb') as f:
            epi2cluster = pickle.load(f, encoding='iso-8859-1')
        epi2cluster = {int(k.split('_')[-1]):v for k, v in epi2cluster.items()}
        return epi2cluster
    if dataset_name == 'reslevel_data':
        cluster_path_dict = {
            'cdr3': os.path.join(DATA_ROOT, 'cluster/reslevel_cdr3_cluster_0.2.pkl'),
            'epi': os.path.join(DATA_ROOT, 'cluster/reslevel_epi_cluster_0.2.pkl'),
        }
        cdr32cluster, epi2cluster = {}, {}
        if ('new_cdr3' in split) or ('both_new' in split):
            with open(cluster_path_dict['cdr3'], 'rb') as f:
                cdr32cluster = pickle.load(f, encoding='iso-8859-1')
        if ('new_epi' in split) or ('both_new' in split):
            with open(cluster_path_dict['epi'], 'rb') as f:
                epi2cluster = pickle.load(f, encoding='iso-8859-1')
        return cdr32cluster, epi2cluster


class ResLevelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        path = config.path
        data = self.load_data(path)
        self.data = self.encoding(data)


    def load_data(self, path):
        df = pd.read_csv(os.path.join(path['summary']))

        cdr3_seqs = df['cdr3'].values
        epi_seqs = df['epitope'].values
        pdb_chains = df['pdb_chains'].values
        pdb_mat = []
        for pdb in pdb_chains:
            df_mat = pd.read_csv(os.path.join(path['mat'], pdb + '.csv'), index_col=0)
            pdb_mat.append(df_mat.values)
        return {
            'cdr3': cdr3_seqs,
            'epi': epi_seqs,
            'dist_mat': pdb_mat,
            'pdb_chains': pdb_chains,
        }

    def encoding(self, data):
        cdr3, epi, mat = data['cdr3'], data['epi'], data['dist_mat']
        # with Pool(processes=64) as p:
        #     enc_cdr3 = p.map(encoding_cdr3_single, cdr3)
        #     enc_epi = p.map(encoding_epi_single, epi)
        enc_cdr3 = encoding_cdr3(cdr3)
        enc_epi = encoding_epi(epi)
        enc_dist_mat, masking = encoding_dist_mat(mat)
        enc_contact_mat = np.int64(enc_dist_mat < 5.)

        data['cdr3'] = np.array(enc_cdr3)
        data['epi'] = np.array(enc_epi)
        data['dist_mat'] = np.array(enc_dist_mat)
        data['mask_mat'] = np.array(masking)
        data['contact_mat'] = np.array(enc_contact_mat)
        data['cdr3_seqs'] = cdr3
        data['epi_seqs'] = epi

        return data

    def __len__(self):
        return len(self.data['dist_mat'])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}

        
    def get_all_values(self, key):
        return self.data[key]

    # def get_max_dist(self):
    #     return np.max(self.data['dist_mat'], axis=0)


class SeqLevelDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        path = config.path
        file_list = config.file_list
        data = self.load_data(path, file_list)
        negative = getattr(config, 'negative', 'original')
        if negative == 'original':
            print('original negative samples')
        elif negative == 'shuffle':
            print('shuffle negative samples')
            data = self.make_shuffled_nega(data)
        self.data = self.encoding(data)

        # add baseline encoding
        baseline = getattr(config, 'baseline', None)
        if baseline is not None:
            print('Add baseline encoding', baseline)
            self.data = self.add_encoding(self.data, baseline)

    def add_encoding(self, data, baseline):
        return data


    def load_data(self, path, file_list):
        for file in file_list:
            try:
                df = pd.read_csv(os.path.join(path, file+'.tsv'), sep='\t')
                data_this = df[['cdr3', 'epitope', 'label']].values
            except FileNotFoundError:
                df = pd.read_csv(os.path.join(path, file+'.csv'))
                data_this = df[['cdr3', 'epi', 'y_true']].values

            if 'data' not in locals():
                data = [[] for _ in range(len(data_this[0]))]
            for i in range(len(data_this[0])):
                data[i].extend(data_this[:, i])
        
        # load epitope id
        try:
            df_epi = pd.read_csv(os.path.join(path, 'positive_epi_dist.tsv'), sep='\t', index_col=0)
            df_epi = df_epi['Epitope_idx']
            epi_ids = df_epi.loc[data[1]].values
        except FileNotFoundError:
            print('epi ID if not found!')
            epi_ids = - np.ones(len(data[1]))
        data = data + [epi_ids]

        # to dict
        data = {
            'cdr3': np.array(data[0]),
            'epi': np.array(data[1]),
            'labels': np.array(data[2]),
            'epi_id': np.array(data[3])
        }
        return data

    def encoding(self, data):
        cdr3, epi = data['cdr3'], data['epi']
        with Pool(processes=64) as p:
            enc_cdr3 = p.map(encoding_cdr3_single, cdr3)
            enc_epi = p.map(encoding_epi_single, epi)
        # enc_cdr3 = encoding_cdr3(cdr3)
        # enc_epi = encoding_epi(epi)
        data['cdr3'] = np.array(enc_cdr3)
        data['epi'] = np.array(enc_epi)
        data['cdr3_seqs'] = cdr3
        data['epi_seqs'] = epi
        return data

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        return {key:value[idx] for key, value in self.data.items()}
        # return {
        #     'cdr3': self.data['cdr3'][idx],
        #     'epi': self.data['epi'][idx],
        #     'labels': self.data['labels'][idx],
        #     'epi_id': self.data['epi_id'][idx],
        #     'cdr3_seqs': self.data['cdr3_seqs'][idx],
        #     'epi_seqs': self.data['epi_seqs'][idx],
        # }

    def get_all_epiid(self):
        return self.data['epi_id']

    def make_shuffled_nega(self, data):
        cdr3 = data['cdr3']
        epi = data['epi']
        labels = data['labels']
        epi_id = data['epi_id']
        # get out positive samples  
        ind_pos = (labels == 1)
        cdr3_pos = cdr3[ind_pos]
        epi_pos = epi[ind_pos]
        labels_pos = labels[ind_pos]
        epi_id_pos = epi_id[ind_pos]
        # prepare info
        ratio = 5  # (len(ind_pos) - ind_pos.sum()) /ind_pos.sum()
        df_pos = pd.DataFrame({'cdr3': cdr3_pos, 'epi': epi_pos, 'labels':labels_pos, 'epi_id':epi_id_pos})
        epi2id = df_pos[['epi', 'epi_id']].set_index('epi').to_dict()['epi_id']

        # make negative by shuffling
        df_all = pd.pivot(df_pos, index='epi', columns='cdr3', values='labels')
        df_all = df_all.fillna(0)
        df_all = df_all.unstack().reset_index().rename(columns={0: 'labels'})
        df_all_shuffled = df_all[df_all['labels']==0]
        epi_counts = df_pos['epi'].value_counts()
        # sample negatives
        df_new_neg_list = []
        for epi, counts_pos in tqdm(epi_counts.items(), total=len(epi_counts), desc='sampling negatives'):
            try:
                df_new_neg = df_all_shuffled[df_all_shuffled['epi']==epi].sample(int(counts_pos * ratio))
            except ValueError:
                df_new_neg = df_all_shuffled[df_all_shuffled['epi']==epi].sample(int(counts_pos * ratio), replace=True)
            df_new_neg_list.append(df_new_neg)
        df_new_neg = pd.concat(df_new_neg_list)
        cdr3_neg = df_new_neg['cdr3']
        epi_neg = df_new_neg['epi']
        labels_neg = df_new_neg['labels']
        epi_id_neg = np.array([epi2id[epi] for epi in epi_neg])
        # concat
        cdr3_all = np.concatenate([cdr3_pos, cdr3_neg], axis=0)
        epi_all = np.concatenate([epi_pos, epi_neg], axis=0)
        labels_all = np.concatenate([labels_pos, labels_neg], axis=0)
        epi_id_all = np.concatenate([epi_id_pos, epi_id_neg], axis=0)
        return {
            'cdr3': cdr3_all,
            'epi': epi_all,
            'labels': labels_all,
            'epi_id': epi_id_all,
        }


class AllEpiDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        path = config.path
        seqs = self.load_data(path)
        self.data = self.encoding(seqs)

    def load_data(self, path):
        df = pd.read_csv(path)
        # drop duplicates and non-standard
        seqs = df['seqs'].values
        seqs = np.unique(seqs)
        seqs = np.array([seq for seq in seqs if np.all([aa in tokenizer.res_all for aa in seq])])
        np.random.shuffle(seqs)
        return seqs

    def encoding(self, seqs):
        encoding = np.zeros([len(seqs), 12], dtype='long')
        for i, seq in tqdm(enumerate(seqs), desc='Encoding seqs', total=len(seqs)):
            len_seq = len(seq)
            if len_seq == 8:
                encoding[i, 2:len_seq+2] = tokenizer.id_list(seq)
            elif (len_seq == 9) or (len_seq == 10):
                encoding[i, 1:len_seq+1] = tokenizer.id_list(seq)
            else:
                encoding[i, :len_seq] = tokenizer.id_list(seq)
        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ...]