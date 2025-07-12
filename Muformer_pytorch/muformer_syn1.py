import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from Muformer_pytorch.rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding
import numpy as np
# helpers



    
def exists(val):
    return val is not None

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# time token shift

def shift(t, amt):
    if amt == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))

class PreTokenShift(nn.Module):
    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f = f)

        # shift along time frame before and after

        dim_chunk = (dim // 3)
        chunks = x.split(dim_chunk, dim = -1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim = -1)

        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim = 1)
        return self.fn(x, *args, **kwargs)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention

def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)  ## ???

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        
        channel_att_sum = None  
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
        
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        # x torch.Size([1, 151, 1800])
        # 'b f c (hp p1) (wp p2) -> b (f hp wp) (p1 p2 c)' 无所谓时刻，对不同patch之间进行计算注意力
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # torch.Size([1, 151, 480]) torch.Size([1, 151, 480]) torch.Size([1, 151, 480])
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))  # 分到多个头下
        # [1, 151, 8 * 60] -> [8, 151, 60]
        q = q * self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v, mask = cls_mask)

        # rearrange across time or space
        # time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb)
        # spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
        # spectral_attn(x, 'b (f n) d', '(b)', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
        # n:每一个时相的patch数
        # f:时相数
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))
        
        # add rotary embeddings, if applicable
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)
        
        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))
        # print(r)
        # print(cls_k.shape)
        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        out = attn(q_, k_, v_, mask = mask)
        
        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        # print('attn', out.shape)
        # concat back the cls token
        out = torch.cat((cls_out, out), dim = 1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

# main classes
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out
    
class Muformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_frames,
        P,
        image_width,
        image_height,
        patch_size,
        channels,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        rotary_emb = True,
        shift_tokens = False
    ):
        super().__init__()
        assert (image_height * image_width) % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_width // patch_size) * (image_height // patch_size)
        num_positions = num_frames * num_patches

        self.P = P
        self.dim = dim 
        self.channels = channels
        self.num_frames = num_frames
        self.div_dim = int(dim / num_frames)
        self.size = image_height * image_width
        self.image_height = image_height
        self.image_width = image_width
        self.heads = heads
        self.patch_size = patch_size
        self.abu_order = (0, 1, 2)
        # self.to_patch_embedding = nn.Linear(patch_dim, P * dim)
        self.cls_token = nn.Parameter(torch.randn(1, P * dim))
        self.order = (0, 1, 2)
        self.b_norm = nn.BatchNorm2d(channels, momentum=0.9)
        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, P * dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.Dropout(0.15),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*P)//patch_size**2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d((dim*P)//patch_size**2, momentum=0.5),
            nn.LeakyReLU()
        )
        
        self.res_block1 = nn.Sequential(
            nn.Conv2d(18, 18, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(18, momentum=0.9),
            nn.LeakyReLU()
        )
        
        self.res_block2 = nn.Sequential(
            nn.Conv2d(18, 18, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(18, momentum=0.9),
            nn.LeakyReLU()
        )
        
        
        self.conv2d = nn.Sequential(
            nn.Conv2d(18, 18, kernel_size=(1, 1), stride=(1, 1), padding=0),
        )
        
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(P * dim , dropout = ff_dropout)
            time_attn = Attention(P * dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            spatial_attn = Attention(P * dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
            # spectral_attn = Attention(P * dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)

            if shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(P * dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.upscale = nn.Sequential(
            # nn.LayerNorm(self.div_dim),
            nn.Linear(self.div_dim, self.size ),
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(P),
            nn.Softmax(dim=1),
        )
        
        self.decoder1 = nn.Sequential(
            nn.Linear(P, channels, bias=False)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(P, channels, bias=False)
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(P, channels, bias=False)
        )
        self.decoder4 = nn.Sequential(
            nn.Linear(P, channels, bias=False)
        )
        self.decoder5 = nn.Sequential(
            nn.Linear(P, channels, bias=False)
        )
        self.decoder6 = nn.Sequential(
            nn.Linear(P, channels, bias=False)
        )
        

        self.to_latent = nn.Identity()
        
        self.cbam = CBAM(gate_channels=18, reduction_ratio=4, pool_types=['avg', 'max'])
        
    def change_enhancement_module(self, HSI_emb):
        HSI = HSI_emb.unsqueeze(0)

        HSI0_conv = self.res_block1(HSI[:, 0])
        HSI1_conv = self.res_block1(HSI[:, 1])
        HSI2_conv = self.res_block1(HSI[:, 2])
        HSI3_conv = self.res_block1(HSI[:, 3])
        HSI4_conv = self.res_block1(HSI[:, 4])
        HSI5_conv = self.res_block1(HSI[:, 5])    #(1, 18, 50, 50)
        
        HSI0_conv_1 = self.res_block2(HSI[:, 0])
        HSI1_conv_1 = self.res_block2(HSI[:, 1])
        HSI2_conv_1 = self.res_block2(HSI[:, 2])
        HSI3_conv_1 = self.res_block2(HSI[:, 3])
        HSI4_conv_1 = self.res_block2(HSI[:, 4])
        HSI5_conv_1 = self.res_block2(HSI[:, 5])    #(1, 18, 50, 50)
        
        M1 = HSI1_conv - HSI[0, 0].unsqueeze(0)
        M2 = HSI2_conv - HSI[0, 1].unsqueeze(0)
        M3 = HSI3_conv - HSI[0, 2].unsqueeze(0)
        M4 = HSI4_conv - HSI[0, 3].unsqueeze(0)
        M5 = HSI5_conv - HSI[0, 4].unsqueeze(0)
        M6 = torch.zeros_like(HSI5_conv)
        
        M1_1 = HSI1_conv_1 - HSI[0, 0].unsqueeze(0)
        M2_1 = HSI2_conv_1 - HSI[0, 1].unsqueeze(0)
        M3_1 = HSI3_conv_1 - HSI[0, 2].unsqueeze(0)
        M4_1 = HSI4_conv_1 - HSI[0, 3].unsqueeze(0)
        M5_1 = HSI5_conv_1 - HSI[0, 4].unsqueeze(0)
        M6_1 = torch.zeros_like(HSI5_conv_1)
        
        
        M = torch.cat([M1, M2, M3, M4, M5, M6], dim=0)  #(6, 18, 50, 50)
        M_t = torch.cat([M1_1, M2_1, M3_1, M4_1, M5_1, M6_1], dim=0)
        
        A1 = torch.sigmoid(self.conv2d(M))   
        A2 = torch.sigmoid(self.conv2d(M_t))
        
        A1 = rearrange(A1, 'f c (hp p1) (wp p2) -> (f hp wp) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        A2 = rearrange(A2, 'f c (hp p1) (wp p2) -> (f hp wp) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        gap_pooling = nn.AdaptiveAvgPool1d(1)
        A1 = gap_pooling(A1.transpose(1, 0))
        A1 = torch.transpose(A1, 1, 0)
        A2 = gap_pooling(A2.transpose(1, 0))
        A2 = torch.transpose(A2, 1, 0)
        return HSI, A1, A2
        
        
    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
        
    def forward(self, Mu_HSI, mask = None):
        # Mu_HSI.shape torch.Size([1, 6, 173, 150, 110])
        b, f, _, h, w, *_, device, p = *Mu_HSI.shape, Mu_HSI.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of Mu_HSI must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)

        hp, wp = (h // p), (w // p)
        n = hp * wp
        HSI_test = Mu_HSI.contiguous().view(6, self.channels, self.image_height, self.image_width)
        HSI_emb = self.encoder(HSI_test)
        HSI, A1, A2 = self.change_enhancement_module(HSI_emb)
        
        HSI_re = rearrange(HSI, 'b f c (hp p1) (wp p2) -> b (f hp wp) (p1 p2 c)', p1 = p, p2 = p)  # torch.Size([1, 990, 1800])
                                                                                                  # torch.Size([1, 150, 1800])

        cls_token = repeat(self.cls_token, 'n d -> b n d', b = b)
        x =  torch.cat((cls_token, HSI_re), dim = 1)

        # positional embedding
        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device = device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device = device)
            image_pos_emb = self.image_rot_emb(hp, wp, device = device)
            
        # calculate masking for uneven number of frames
        frame_mask = None
        cls_attn_mask = None
        if exists(mask):
            mask_with_cls = F.pad(mask, (1, 0), value = True)
            frame_mask = repeat(mask_with_cls, 'b f -> (b h n) () f', n = n, h = self.heads)

            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n = n, h = self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value = True)

        # time and space attention
        # x torch.Size([1, 151, 1800])
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = frame_mask, cls_mask = cls_attn_mask, rot_emb = frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            # x = spectral_attn(x, '(b h) n d', '(b h) d n', h = self.heads, cls_mask = cls_attn_mask, rot_emb = image_pos_emb) + x
            x = ff(x) + x
        
        # x shape (1, 151, 1800) -> (1, 151, 18, 10, 10)
        # print(x.shape)
        x = x.squeeze(0).view(151, 18, 10, 10)
        x = self.cbam(x)
        x = x.contiguous().view(1, 151, 1800)
        
        cls_token = x[:, 0]
        
        cls_token = cls_token + 5 * (A1 * A2) * cls_token

        out = self.to_latent(cls_token)   # torch.Size([1, 1800])
        out = out.contiguous().view(6, self.P, -1)  # torch.Size([6, 3, 100])
        abu = self.upscale(out).clamp_(0,1).contiguous().view(6, self.P, self.image_height, self.image_width)

        
        
        abu = self.smooth(abu)
        abu_t = abu.reshape(6, self.P, self.image_height*self.image_width)
        out1 = self.decoder1(torch.transpose(abu_t[0], 1, 0))
        out2 = self.decoder2(torch.transpose(abu_t[1], 1, 0))
        out3 = self.decoder3(torch.transpose(abu_t[2], 1, 0))
        out4 = self.decoder4(torch.transpose(abu_t[3], 1, 0))
        out5 = self.decoder5(torch.transpose(abu_t[4], 1, 0))
        out6 = self.decoder6(torch.transpose(abu_t[5], 1, 0))
        
        re_out = torch.stack([out1, out2, out3, out4, out5, out6])
        re_out = re_out.permute(0, 2, 1).reshape(6, self.channels, self.image_height, self.image_width)
        return abu, re_out, out
