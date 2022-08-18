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
from scripts.baselines import ImRexEncoder, NetTCREncoder, ERGOEncoder



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


