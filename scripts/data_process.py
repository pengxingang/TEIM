import os
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
import torch
import sys
sys.path.append('.')
from models import AutoEncoder
base_data_dir = './all_data/'


def GetBlosumMat(residues_list):
    n_residues = len(residues_list)  # the number of amino acids _ 'X'
    blosum62_mat = np.zeros([n_residues, n_residues])  # plus 1 for gap
    bl_dict = substitution_matrices.load('BLOSUM62')
    for pair, score in bl_dict.items():
        if (pair[0] not in residues_list) or (pair[1] not in residues_list):  # special residues not considered here
            continue
        idx_pair0 = residues_list.index(pair[0])  # index of residues
        idx_pair1 = residues_list.index(pair[1])
        blosum62_mat[idx_pair0, idx_pair1] = score
        blosum62_mat[idx_pair1, idx_pair0] = score
    return blosum62_mat


class Tokenizer:
    def __init__(self,):
        self.res_all = ['G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y', 'D', 'N',
                     'E', 'K', 'Q', 'M', 'S', 'T', 'C', 'P', 'H', 'R'] #+ ['X'] #BJZOU
        self.tokens = ['-'] + self.res_all # '-' for padding encoding

    def tokenize(self, index): # int 2 str
        return self.tokens[index]

    def id(self, token): # str 2 int
        try:
            return self.tokens.index(token.upper())
        except ValueError:
            print('Error letter in the sequences:', token)
            if str.isalpha(token):
                return self.tokens.index('X')

    def tokenize_list(self, seq):
        return [self.tokenize(i) for i in seq]

    def id_list(self, seq):
        return [self.id(s) for s in seq]

    def embedding_mat(self):
        blosum62 = GetBlosumMat(self.res_all)
        mat = np.eye(len(self.tokens))
        mat[1:len(self.res_all) + 1, 1:len(self.res_all) + 1] = blosum62
        return mat


def get_numbering(seqs, ):
    """
    get the IMGT numbering of CDR3 with ANARCI tool
    """
    template = ['GVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGTTDQGEVPNGYNVSRSTIEDFPLRLLSAAPSQTSVYF', 'GEGSRLTVL']
    # # save fake tcr file
    save_path = 'tmp_faketcr.fasta'
    id_list = []
    seqs_uni = np.unique(seqs)
    with open(save_path, 'w+') as f:
        for i, seq in enumerate(seqs_uni):
            f.write('>'+str(i)+'\n')
            id_list.append(i)
            total_seq = ''.join([template[0], seq ,template[1]])
            f.write(str(total_seq))
            f.write('\n')
    print('Save fasta file to '+save_path + '\n Aligning...')
    df_seqs = pd.DataFrame(list(zip(id_list, seqs_uni)), columns=['Id', 'cdr3'])
    
    # # using ANARCI to get numbering file
    cmd = ("conda run -n teim "  # this environment name should be the same as the one you install anarci
            " ANARCI"
            " -i tmp_faketcr.fasta  -o tmp_align --csv -p 24")
    res = os.system(cmd)
    
    # # parse numbered seqs data
    df = pd.read_csv('tmp_align_B.csv')
    cols = ['104', '105', '106', '107', '108', '109', '110', '111', '111A', '111B', '112C', '112B', '112A', '112', '113', '114', '115', '116', '117', '118']
    seqs_al = []
    for col in cols:
        if col in df.columns:
            seqs_al_curr = df[col].values
            seqs_al.append(seqs_al_curr)
        else:
            seqs_al_curr = np.full([len(df)], '-')
            seqs_al.append(seqs_al_curr)
    seqs_al = [''.join(seq) for seq in np.array(seqs_al).T]
    df_al = df[['Id']]
    df_al['cdr3_align'] = seqs_al
    
    ## merge
    os.remove('tmp_align_B.csv')
    os.remove('tmp_faketcr.fasta')
    df = df_seqs.merge(df_al, how='inner', on='Id')
    df = df.set_index('cdr3')
    return df.loc[seqs, 'cdr3_align'].values


def load_ae_model(tokenizer, path='./ckpt/epi_ae.ckpt'):
    # tokenizer = Tokenizer()
    ## load model
    model_args = dict(
        tokenizer = tokenizer,
        dim_hid = 32,
        len_seq = 12,
    )
    model = AutoEncoder(**model_args)
    model.eval()

    ## load weights
    state_dict = torch.load(path)
    state_dict = {k[6:]:v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


class PretrainedEncoder:
    def __init__(self, tokenizer):
        self.ae_model = load_ae_model(tokenizer)
        self.tokenizer = tokenizer

    def encode_pretrained_epi(self, epi_seqs):
        enc = self.infer(epi_seqs)
        enc_vec = enc[2]
        enc_seq = enc[-1]
        return enc_seq, enc_vec
    
    def infer(self, seqs):
        # # seqs encoding
        n_seqs = len(seqs)
        len_seqs = [len(seq) for seq in seqs]
        assert (np.max(len_seqs) <= 12) and (np.min(len_seqs)>=8), ValueError('Lengths of epitopes must be within [8, 12]')
        encoding = np.zeros([n_seqs, 12], dtype='int32')
        for i, seq in enumerate(seqs):
            len_seq = len_seqs[i]
            if len_seq == 8:
                encoding[i, 2:len_seq+2] = self.tokenizer.id_list(seq)
            elif (len_seq == 9) or (len_seq == 10):
                encoding[i, 1:len_seq+1] = self.tokenizer.id_list(seq)
            else:
                encoding[i, :len_seq] = self.tokenizer.id_list(seq)
        # # pretrained ae features
        inputs = torch.from_numpy(encoding)
        out, seq_enc, vec, indices = self.ae_model(inputs)
        out = np.argmax(out.detach().cpu().numpy(), -1)
        return [
            out,
            seq_enc.detach().cpu().numpy(),
            vec.detach().cpu().numpy(),
            indices,
            encoding
        ]


def encode_cdr3(cdr3, tokenizer):
    len_cdr3 = [len(s) for s in cdr3]
    max_len_cdr3 = np.max(len_cdr3)
    assert max_len_cdr3 <= 20, 'The cdr3 length must <= 20'
    max_len_cdr3 = 20
    
    seqs_al = get_numbering(cdr3)
    num_samples = len(seqs_al)

    # encoding
    encoding_cdr3 = np.zeros([num_samples, max_len_cdr3], dtype='int32')
    for i, seq in enumerate(seqs_al):
        encoding_cdr3[i, ] = tokenizer.id_list(seq)
    return encoding_cdr3


def encode_epi(epi, tokenizer):
    # tokenizer = Tokenizer()
    encoding_epi = np.zeros([12], dtype='int32')
    len_epi = len(epi)
    if len_epi == 8:
        encoding_epi[2:len_epi+2] = tokenizer.id_list(epi)
    elif (len_epi == 9) or (len_epi == 10):
        encoding_epi[1:len_epi+1] = tokenizer.id_list(epi)
    else:
        encoding_epi[:len_epi] = tokenizer.id_list(epi)
    return encoding_epi

