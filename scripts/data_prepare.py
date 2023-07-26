from multiprocessing import Pool
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import Subset
from utils.encoding import *


def make_shuffled_nega(cdr3, epi, labels, epi_id):

    # get out positive samples  
    ind_pos = (labels == 1)
    cdr3_pos = cdr3[ind_pos]
    epi_pos = epi[ind_pos]
    labels_pos = labels[ind_pos]
    epi_id_pos = epi_id[ind_pos]
    # prepare info
    ratio = (len(ind_pos) - ind_pos.sum()) /ind_pos.sum()
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


def get_cluster(dataset_name, split):
    if (dataset_name in ['seqlevel_data']) and split == 'cv-new_epitope':
        cluster_path = './data/cluster/seqlevel_epi_cluster_0.5.pkl'
        with open(cluster_path, 'rb') as f:
            epi2cluster = pickle.load(f, encoding='iso-8859-1')
        epi2cluster = {int(k.split('_')[-1]):v for k, v in epi2cluster.items()}
        return epi2cluster
    elif dataset_name == 'reslevel_data':
        cluster_path_dict = {
            'cdr3': './data/cluster/reslevel_cdr3_cluster_0.2.pkl',
            'epi': './data/cluster/reslevel_epi_cluster_0.2.pkl',
        }
        cdr32cluster, epi2cluster = {}, {}
        if ('new_cdr3' in split) or ('both_new' in split):
            with open(cluster_path_dict['cdr3'], 'rb') as f:
                cdr32cluster = pickle.load(f, encoding='iso-8859-1')
        if ('new_epi' in split) or ('both_new' in split):
            with open(cluster_path_dict['epi'], 'rb') as f:
                epi2cluster = pickle.load(f, encoding='iso-8859-1')
    else:
        raise ValueError('dataset_name {} not supported'.format(dataset_name))
    return cdr32cluster, epi2cluster


def get_split(split_type, dataset_name, all_cdr3, all_epi):
    cdr32cluster, epi2cluster = get_cluster(dataset_name, split_type)

    assert split_type in ['cv-both_new', 'cv-new_cdr3', 'cv-new_epi']
    kfold = GroupKFold(3) if split_type == 'cv-both_new' else GroupKFold(5)
    if len(cdr32cluster) != 0:  # new_cdr3, both_new
        cdr3_group_id = [cdr32cluster[cdr3_this] for cdr3_this in all_cdr3]
        split_cdr3 = list(kfold.split(all_cdr3, groups=cdr3_group_id))
        split = split_cdr3
    if len(epi2cluster) != 0:  # new_epi, both_new
        epi_group_id = [epi2cluster[epi_this] for epi_this in all_epi]
        split_epi = list(kfold.split(all_epi, groups=epi_group_id))
        split = split_epi
    if split_type == 'cv-both_new':
        split = [[np.intersect1d(fold_tcr[0], fold_epi[0]), np.intersect1d(fold_tcr[1], fold_epi[1])]
            for fold_tcr, fold_epi in zip(split_cdr3, split_epi)]
    
    return split
    