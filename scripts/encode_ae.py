import os 
import numpy as np
import pandas as pd

base_data_dir = './all_data'
mode_weights_dir = './epi_pretrain/ae'
try:
    from epi_pretrain.ae.fit_ae import *
except:
    import sys
    sys.path.append('.')
    from epi_pretrain.ae.fit_ae import *



def infer(model, seqs):
    ## encode data
    n_seqs = len(seqs)
    tokenizer = Tokenizer()
    encoding = np.zeros([len(seqs), 12], dtype='int32')
    for i, seq in enumerate(seqs):
        len_seq = len(seq)
        if len_seq == 8:
            encoding[i, 2:len_seq+2] = tokenizer.id_list(seq)
        elif (len_seq == 9) or (len_seq == 10):
            encoding[i, 1:len_seq+1] = tokenizer.id_list(seq)
        else:
            encoding[i, :len_seq] = tokenizer.id_list(seq)
            
    
    ## predict 
    inputs = torch.from_numpy(encoding)
    out, seq_enc, vec, indices = model(inputs)

    ## evaludate
    out = np.argmax(out.detach().cpu().numpy(), -1)
    # for i, (x, y_pred) in enumerate(zip(encoding, out)):
    #     print()
    #     print(x)
    #     print(y_pred)

    return [
        out,
        seq_enc.detach().cpu().numpy(),
        vec.detach().cpu().numpy(),
        indices,
        encoding
    ]

def main(data_mode, name):
    ## load model
    model = load_ae_model(data_mode, name)

    ## load data
    df = pd.read_csv('./all_data/raw_data/TCR_0127/positive_epi_dist.tsv', sep='\t')
    seqs = list(df['Epitope'].values)
    
    out, seq_enc, vec, indices, encoding = infer(model, seqs)

    ## evaludate
    # out = np.argmax(out, -1)
    for i, (x, y_pred) in enumerate(zip(encoding, out)):
        print(x)
        print(y_pred)
        print()

    ## save
    save_path_func = lambda x: os.path.join(base_data_dir, 'pretrained_data', 'ae_'+data_mode['seq2vec'], x)
    np.save(save_path_func('seqs'), np.array(seqs))
    np.save(save_path_func('seq_enc'), seq_enc)
    np.save(save_path_func('vec'), vec)
    try:
        np.save(save_path_func('indices'), indices)
    except:
        pass
    print()

if __name__ == '__main__':
    data_mode = {
        'seq2vec':'flatten'
    }
    name='1365_last'
    main(data_mode, name)