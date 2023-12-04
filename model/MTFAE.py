import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AttentionLayer
from .embed import DataEmbedding, PositionalEmbedding

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, T, D]
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)

        if self.norm is not None:
            x = self.norm(x)

        return x, attlist

class FreEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,d_model,1, dtype=torch.cfloat))

        self.fr = fr
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]

        # converting to frequency domain and calculating the mag
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]

        # masking smaller mag
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag<quantile)
        cx[mag<quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:,0],idx[:,1],idx[:,2]]

        # converting to time domain
        ix = torch.fft.irfft(cx).transpose(1,2)

        # encoding tokens
        dx, att = self.enc(ix)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]
    
class TemEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.dec = Encoder(
            [
                    AttentionLayer( d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        filters = torch.ones(1,1,self.seq_size).to(device)
        ex2 = ex ** 2

        # calculating summation of ex and ex2
        ltr = F.conv1d(ex.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr[:,:,self.seq_size-1:] /= self.seq_size
        ltr2 = F.conv1d(ex2.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr2[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr2[:,:,self.seq_size-1:] /= self.seq_size
        
        # calculating mean and variance
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        score = ltrd.sum(-1) / ltrm.sum(-1)

        # mask time points
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], (-1*score).topk(x.shape[1]-self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:,None],unmasked_idx,:]
        
        # encoding unmasked tokens and getting masked tokens
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx = masked_idx)
        
        tokens = torch.zeros(ex.shape,device=device)
        tokens[torch.arange(ex.shape[0])[:,None],unmasked_idx,:] = ux
        tokens[torch.arange(ex.shape[0])[:,None],masked_idx,:] = masked_tokens

        # decoding tokens
        dx, att = self.dec(tokens)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]


class MTFA(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(MTFA, self).__init__()
        global device
        device = dev
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, fr)

    def forward(self, x):
        # x: [B, T, C]
        tematt = self.tem(x) # tematt: [B, T, T]
        freatt = self.fre(x) # freatt: [B, T, T]
        return tematt, freatt