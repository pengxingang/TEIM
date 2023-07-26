import numpy as np
from tqdm import tqdm
from Bio.Align import substitution_matrices
import sys
sys.path.append('.')


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

tokenizer = Tokenizer()


def encoding_epi(seqs, max_len=12):
    encoding = np.zeros([len(seqs), max_len], dtype='long')
    for i, seq in tqdm(enumerate(seqs), desc='Encoding epi seqs', total=len(seqs)):
        len_seq = len(seq)
        if len_seq == 8:
            encoding[i, 2:len_seq+2] = tokenizer.id_list(seq)
        elif (len_seq == 9) or (len_seq == 10):
            encoding[i, 1:len_seq+1] = tokenizer.id_list(seq)
        else:
            encoding[i, :len_seq] = tokenizer.id_list(seq)
    return encoding

def encoding_cdr3(seqs, max_len=20):
    encoding = np.zeros([len(seqs), max_len], dtype='long')
    for i, seq in tqdm(enumerate(seqs), desc='Encoding cdr3s', total=len(seqs)):
        len_seq = len(seq)
        i_start =  max_len // 2 - len_seq // 2
        encoding[i, i_start:i_start+len_seq] = tokenizer.id_list(seq)
    return encoding

def encoding_cdr3_single(seq, max_len=20):
    encoding = np.zeros(max_len, dtype='long')
    len_seq = len(seq)
    i_start =  max_len // 2 - len_seq // 2
    encoding[i_start:i_start+len_seq] = tokenizer.id_list(seq)
    return encoding

def encoding_epi_single(seq, max_len=12):
    encoding = np.zeros(max_len, dtype='long')
    len_seq = len(seq)
    if len_seq == 8:
        encoding[2:len_seq+2] = tokenizer.id_list(seq)
    elif (len_seq == 9) or (len_seq == 10):
        encoding[1:len_seq+1] = tokenizer.id_list(seq)
    else:
        encoding[:len_seq] = tokenizer.id_list(seq)
    return encoding


def encoding_dist_mat(mat_list, max_cdr3=20, max_epi=12):
    encoding = np.zeros([len(mat_list), max_cdr3, max_epi], dtype='float32')
    masking = np.zeros([len(mat_list), max_cdr3, max_epi], dtype='bool')
    for i, mat in tqdm(enumerate(mat_list), desc='Encoding dist mat', total=len(mat_list)):
        len_cdr3, len_epi = mat.shape
        i_start_cdr3 = max_cdr3 // 2 - len_cdr3 // 2
        if len_epi == 8:
            i_start_epi = 2
        elif (len_epi == 9) or (len_epi == 10):
            i_start_epi = 1
        else:
            i_start_epi = 0
        encoding[i, i_start_cdr3:i_start_cdr3+len_cdr3, i_start_epi:i_start_epi+len_epi] = mat
        masking[i, i_start_cdr3:i_start_cdr3+len_cdr3, i_start_epi:i_start_epi+len_epi] = True
    return encoding, masking


def decoding_one_mat(mat, len_cdr3, len_epi):
    decoding = np.zeros([len_cdr3, len_epi] + list(mat.shape[2:]), dtype=mat.dtype)
    i_start_cdr3 = 10 - len_cdr3 // 2
    if len_epi == 8:
        i_start_epi = 2
    elif (len_epi == 9) or (len_epi == 10):
        i_start_epi = 1
    else:
        i_start_epi = 0
    decoding = mat[i_start_cdr3:i_start_cdr3+len_cdr3, i_start_epi:i_start_epi+len_epi] 
    return decoding




