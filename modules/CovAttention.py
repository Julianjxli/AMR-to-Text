import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
 
 
 
# 提供NoAttention type
class BaseAttention(nn.Module):
    def __init__(self, context_dim):
        super(BaseAttention, self).__init__()
        self.context_dim = context_dim
 
    def forward(self, decoder_state, src_hids, src_lengths):
        '''
        :param decoder_state: bsz * decoder_hidden_state_dim
        :param src_hids: src_len * bsz * context_dim
        :param src_lengths: bsz * 1, actual sequence lens
        :return:
        outputs: bsz * context_dim
        attn_scores: max_src_len * bsz
        '''
        raise NotImplementedError



def Linear(in_features, out_features, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m
 

class Cov_Attention(BaseAttention):

    def __init__(self,  context_dim, dropout=0.1,  **kwargs):
        super().__init__(context_dim)
 
        self.context_dim = context_dim


        self.temperature = context_dim**0.5

        self.dropout = nn.Dropout(dropout)

        self.Linear=nn.Linear(1,1,bias=False)
        

    def forward(self, decoder_state, source_hids, coverage_before, encoder_padding_mask):


        # Reshape to bsz x src_len x context_dim

        source_hids = source_hids.transpose(0, 1)#(bsz,src_len,model_dim)

        attn = torch.matmul(decoder_state / self.temperature, source_hids.transpose(1, 2))#(bsz,tgt_len,model_dim)x(bsz,model_dim,src_len)=(bsz,tgt_len,src_len)

        coverage_infeature=self.Linear(coverage_before).squeeze(3)#[bs, tgt_len, src_len]

        Newattn=attn+coverage_infeature #[bs,tgt_len,scr_len]
        
        encoder_padding_mask=encoder_padding_mask.unsqueeze(1)

        # Mask + softmax 
        if encoder_padding_mask is not None:
            Newattn = Newattn.masked_fill(encoder_padding_mask == 0,torch.tensor(-1e9))
        Newattn=self.dropout(F.softmax(Newattn, dim=-1))


        output=torch.matmul(Newattn,source_hids)#(bsz,tgt_len,model_dim)
        output= self.dropout(output)


        return  output, Newattn