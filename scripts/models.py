import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate as co_fn
# from ..pretrain.models import TransformerVAE
import numpy as np

class ResNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        # self.linear = nn.Linear(cnn.in_channels, cnn.out_channels)
    def forward(self, data):
        tmp_data = self.cnn(data)
        # out = tmp_data + self.linear(data.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = tmp_data + data
        return out

class NewCNN2dBaseline(nn.Module):
    def __init__(self, tokenizer, dim_hidden, model_type, form, **kwargs):
        super().__init__()
        self.model_type = model_type

        ## emb
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0, )

        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hidden, 1,),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
        )

        self.seq_epi =nn.Sequential(
            nn.Conv1d(dim_emb, dim_hidden, 1,),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
        )

        # if 'ae' not in self.model_type:
        #     self.cnn_module = nn.Sequential(
        #         # nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1),
        #         ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
        #         nn.BatchNorm2d(dim_hidden),
        #         nn.ReLU(),

        #         # nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1),
        #         ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
        #         nn.BatchNorm2d(dim_hidden),
        #         nn.ReLU(),

        #         nn.AdaptiveMaxPool2d(1),
        #         nn.Flatten(),
        #     )
        # else:
        self.cnn_list = nn.ModuleList([
            nn.Sequential(
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                nn.BatchNorm2d(dim_hidden),
                nn.ReLU(),
            ),
            nn.ModuleList([
                ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                # nn.Conv2d(dim_hidden + 128, dim_hidden, kernel_size=3, padding=1),
                nn.Sequential(
                    nn.BatchNorm2d(dim_hidden),
                    nn.ReLU(),
                )
            ]),
            nn.Sequential(
                nn.AdaptiveMaxPool2d(1),
                nn.Flatten(),
            ),
        ])  
        self.global_linear = nn.Conv2d(32, dim_hidden, kernel_size=1, stride=1)

            
        dim_out = dim_hidden
        self.dense_module = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_out, 1),
            nn.Sigmoid()
        )


    def forward(self, inputs):
        if len(inputs) == 3:
            cdr3, epi, pretrained = inputs
        else:
            cdr3, epi = inputs
        # batch_size = len(cdr3)
        if len(cdr3.shape) == 2:
            cdr3_aa, epi_aa = cdr3.long(), epi.long()
        elif len(cdr3.shape) == 3:
            cdr3_aa, epi_aa = cdr3[..., 0].long(), epi[..., 0].long()
            cdr3_bio, epi_bio = cdr3[..., 1:].float(), epi[..., 1:].float()
        len_cdr3, len_epi = cdr3_aa.shape[1], epi_aa.shape[1]

        ## embedding
        cdr3_emb = self.embedding_module(cdr3_aa)
        epi_emb = self.embedding_module(epi_aa)

        ## integrate pretrained
        if 'ae' in self.model_type:
            epi_vec = pretrained #+ epi_emb

        ## sequence features
        cdr3_feat = self.seq_cdr3(cdr3_emb.transpose(1, 2))
        epi_feat = self.seq_epi(epi_emb.transpose(1, 2))

        ## out module
        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])
        epi_feat_mat = epi_feat.unsqueeze(2).repeat([1, 1, len_cdr3, 1])
        inter_map = cdr3_feat_mat * epi_feat_mat
        # inter_map = torch.cat([cdr3_feat_mat, epi_feat_mat], axis=1)

        if 'ae' not in self.model_type:
            cnn_out = self.cnn_module(inter_map)
        else:
            cnn_out = inter_map
            vec = pretrained.unsqueeze(2).unsqueeze(3)
            n_layers = len(self.cnn_list)
            for i in range(n_layers - 1):
                if i == n_layers - 2: # the last cnn layer
                    global_feat = self.global_linear(vec)
                    inter_map = self.cnn_list[i][0](inter_map)
                    inter_map = inter_map + global_feat
                    inter_map = self.cnn_list[i][1](inter_map)
                    # inter_map = torch.cat([inter_map, global_feat.repeat([1, 1, len_cdr3, len_epi])], 1)
                else:
                    inter_map = self.cnn_list[i](inter_map)
            cnn_out = self.cnn_list[-1](inter_map)

        outputs = self.dense_module(cnn_out,)
        self.cnn_out = cnn_out
        self.outputs = outputs
        self.inter_map = inter_map
        return outputs

class AutoEncoder(nn.Module):
    def __init__(self, 
        tokenizer,
        dim_hid,
        len_seq,
    ):
        super().__init__()
        embedding = tokenizer.embedding_mat()
        vocab_size, dim_emb = embedding.shape
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), padding_idx=0, )
        self.encoder = nn.Sequential(
            nn.Conv1d(dim_emb, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        # if data_mode['seq2vec'] == 'pool':
        #     self.seq2vec = nn.MaxPool1d(kernel_size=len_seq, stride=1, return_indices=True)
        #     self.vec2seq = nn.MaxUnpool1d(kernel_size=len_seq, stride=1)
        # elif data_mode['seq2vec'] == 'flatten':
        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len_seq * dim_hid, dim_hid),
            nn.ReLU()
        )
        self.vec2seq = nn.Sequential(
            nn.Linear(dim_hid, len_seq * dim_hid),
            nn.ReLU(),
            View(dim_hid, len_seq)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim_hid),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hid, vocab_size)

    def forward(self, inputs):
        inputs = inputs.long()
        seq_emb = self.embedding_module(inputs)
        
        seq_enc = self.encoder(seq_emb.transpose(1, 2))
        # if self.data_mode['seq2vec'] == 'pool':
        #     vec, indices = self.seq2vec(seq_enc)
        #     seq_repr = self.vec2seq(vec, indices)
        #     vec, indices = vec.squeeze(-1), indices.squeeze(-1)
        # elif self.data_mode['seq2vec'] == 'flatten':
        vec = self.seq2vec(seq_enc)
        seq_repr = self.vec2seq(vec)
        indices = None
        seq_dec = self.decoder(seq_repr)
        out = self.out_layer(seq_dec.transpose(1, 2))
        return out, seq_enc, vec, indices

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        shape = [input.shape[0]] + list(self.shape)
        return input.view(*shape)