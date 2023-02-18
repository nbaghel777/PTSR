import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn as nn


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1)) #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads = 4, dim_head = None, discriminator = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias = True)

        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            try:
                #print('qkvshape',self.to_qkv.weight)
                u, s, v = torch.svd(self.to_qkv.weight+0.0001)
            except:
                # torch.svd may have convergence issues for GPU and CPU.
                #self.to_qkv.weight=self.to_qkv.weight + 0.001*torch.rand(1152, 384).to("cuda")
                u, s, v = torch.svd(self.to_qkv.weight + 0.001*torch.rand(1152, 384).to("cuda"))
            self.to_qkv.weight = torch.nn.Parameter(self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k = 3, h = self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p = 2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim = -1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score, v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return self.w_out(result)


class DEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(DEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head, discriminator = True)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.dropout(self.attn(x1))
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x


class GEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads = 4, dim_head = None,
        dropout = 0., mlp_ratio = 4):
        super(GEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = SLN(dim)
        self.norm2 = SLN(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout = dropout)

    def forward(self, hl, x):
        hl_temp = self.dropout(self.attn(self.norm1(hl, x))) + hl
        hl_final = self.mlp(self.norm2(hl_temp, x)) + hl_temp
        return x, hl_final


class GTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(GTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(GEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, hl,x):
        for block in self.blocks:
            x, hl = block(hl, x)
            #print(x.shape, hl.shape,)
        #print('block end')
        return x, hl


class DTransformerEncoder(nn.Module):
    def __init__(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        super(DTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head, dropout)

    def _make_layers(self,
        dim,
        blocks = 6,
        num_heads = 8,
        dim_head = None,
        dropout = 0
    ):
        layers = []
        for _ in range(blocks):
            layers.append(DEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """
    def __init__(self, in_features, out_features, bias = True,is_first = False, omega_0 = 30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

import torch.nn.functional as F
import torchvision.transforms as T

def prepatch(img,dpatch_size):
    img_patches = F.unfold(img, kernel_size=dpatch_size,stride=dpatch_size)
    #print(img_patches.shape,'img_patches0')
    img_patches = img_patches.permute(0,2,1)
    #print(img_patches.shape,'img_patches1')
    #img_patches=img_patches.contiguous().view(int(img_patches.shape[0]*(img_patches.shape[1]/(dpatch_size*dpatch_size))), dpatch_size*dpatch_size,img_patches.shape[2])
    #print(img_patches.shape,'img_patches3')
    return img_patches

def postpatch(x,imgshape,dpatch_size,xfactor):
    #x=x.contiguous().view(
    #    imgshape[0],int((x.shape[0]/imgshape[0]) *dpatch_size*xfactor*dpatch_size*xfactor),x.shape[2])
    img_patches=x.permute(0,2,1)
    #print(img_patches.shape,'img_patchespost')
    
    #print([imgshape[-2]*xfactor,imgshape[-1]*xfactor] ,dpatch_size,dpatch_size)
    result = F.fold(img_patches,[imgshape[-2]*xfactor,imgshape[-1]*xfactor] ,dpatch_size,stride=dpatch_size) 
    #print(result.shape,'result')
    #result = T.ColorJitter(brightness=.1, contrast=1, saturation=.3,  hue=0)(result)
    return result

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def bicubic_upsample(x,scale_factor,patchsize):
    H=W=patchsize
    B, N, C = x.size()
    #print('nshape',N)
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=2, mode='bicubic',align_corners=True)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x

def bicubic_downsample(x,scale_factor,patchsize):
    H=W=patchsize
    B, N, C = x.size()
    #print('nshape',N)
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=.5, mode='bicubic',align_corners=True)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x

class Generator(nn.Module):
    def __init__(self,
                ):
        super(Generator, self).__init__()
        self.g1=Generator1()
        self.g2=Generator2()
        self.g3=Generator3()
        self.g4=Generator4()
    def forward(self, img):
        
        # if img.shape[2]==256:
        img1=self.g1(img)
        img1dn = nn.functional.interpolate(img1, scale_factor=.5, mode='bicubic',align_corners=True,recompute_scale_factor=True)
        imgup = nn.functional.interpolate(img, scale_factor=2, mode='bicubic',align_corners=True,recompute_scale_factor=True)
#         else:
#             img1dn=img
#             img1 = nn.functional.interpolate(img, scale_factor=2, mode='bilinear',align_corners=True,recompute_scale_factor=True)
#             imgup = nn.functional.interpolate(img, scale_factor=4, mode='bilinear',align_corners=True,recompute_scale_factor=True)
        
        img2=self.g2(img1dn)
        img2up = nn.functional.interpolate(img2, scale_factor=2, mode='bicubic',align_corners=True,recompute_scale_factor=True)
        img3=self.g3(img2up+img1)
        img3up = nn.functional.interpolate(img3, scale_factor=2, mode='bicubic',align_corners=True,recompute_scale_factor=True)
        
        
        img1up = nn.functional.interpolate(img1, scale_factor=2, mode='bicubic',align_corners=True,recompute_scale_factor=True)
        img4=self.g4(img3up+img1up)
        
        return img4+imgup
    
class Generator1(nn.Module):
    def __init__(self,
        patch_size = 8,
        dim = 192,
        blocks = 5,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Generator1, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.pos_emb1D = nn.Parameter(torch.randn(256 , dim))
        self.Transformer_Encoder1 = GTransformerEncoder(self.dim, blocks, num_heads, dim_head, dropout)
        self.sln_norm = SLN(self.dim)
        
    def forward(self, img):
        x=prepatch(img,self.patch_size)
        x,h= self.Transformer_Encoder1(self.pos_emb1D,x)
        x1 = self.sln_norm(h, x)
        x1=postpatch(x1,img.shape,self.patch_size,xfactor=1)
        return x1
    
    
class Generator2(nn.Module):
    def __init__(self,
        patch_size = 8,
        dim = 192,
        blocks = 5,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Generator2, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.pos_emb1D = nn.Parameter(torch.randn(64, dim))
        self.Transformer_Encoder1 = GTransformerEncoder(self.dim, blocks, num_heads, dim_head, dropout)
        self.sln_norm = SLN(self.dim)
        
    def forward(self, img):
        x=prepatch(img,self.patch_size)
        x,h= self.Transformer_Encoder1(self.pos_emb1D,x)
        x1 = self.sln_norm(h, x)
        x1=postpatch(x1,img.shape,self.patch_size,xfactor=1)
        return x1

class Generator3(nn.Module):
    def __init__(self,
        patch_size = 8,
        dim = 192,
        blocks = 5,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Generator3, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.pos_emb1D = nn.Parameter(torch.randn(256 , dim))
        self.Transformer_Encoder1 = GTransformerEncoder(self.dim, blocks, num_heads, dim_head, dropout)
        self.sln_norm = SLN(self.dim)
        
    def forward(self, img):
        x=prepatch(img,self.patch_size)
        x,h= self.Transformer_Encoder1(self.pos_emb1D,x)
        x1 = self.sln_norm(h, x)
        x1=postpatch(x1,img.shape,self.patch_size,xfactor=1)
        return x1

class Generator4(nn.Module):
    def __init__(self,
        patch_size = 8,
        dim = 192,
        blocks = 5,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Generator4, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.pos_emb1D = nn.Parameter(torch.randn(1024, dim))
        self.Transformer_Encoder1 = GTransformerEncoder(self.dim, blocks, num_heads, dim_head, dropout)
        self.sln_norm = SLN(self.dim)
        
    def forward(self, img):
        x=prepatch(img,self.patch_size)
        x,h= self.Transformer_Encoder1(self.pos_emb1D,x)
        x1 = self.sln_norm(h, x)
        x1=postpatch(x1,img.shape,self.patch_size,xfactor=1)
        return x1
    
class Discriminator(nn.Module):
    def __init__(self,
        in_channels = 3,
        patch_size = 8,
        extend_size = 2,
        dim = 384,
        blocks = 2,
        num_heads = 6,
        dim_head = None,
        dropout = 0
    ):
        super(Discriminator, self).__init__()
        self.patch_size = patch_size + 2 * extend_size
        self.token_dim = in_channels * (self.patch_size ** 2)
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_emb1D = nn.Parameter(torch.randn(self.token_dim + 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
        self.Transformer_Encoder = DTransformerEncoder(dim, blocks, num_heads, dim_head, dropout)

    def forward(self, img):
        # Generate overlappimg image patches
        stride_h = (img.shape[2] - self.patch_size) // 8 + 1
        stride_w = (img.shape[3] - self.patch_size) // 8 + 1
        img_patches = img.unfold(2, self.patch_size, stride_h).unfold(3, self.patch_size, stride_w)
        img_patches = img_patches.contiguous().view(
            img_patches.shape[0], img_patches.shape[2] * img_patches.shape[3], img_patches.shape[1] * img_patches.shape[4] * img_patches.shape[5]
        )
        img_patches = self.project_patches(img_patches)
        batch_size, tokens, _ = img_patches.shape

        # Prepend the classifier token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        img_patches = torch.cat((cls_token, img_patches), dim = 1)

        # Plus the positional embedding
        img_patches = img_patches + self.pos_emb1D[: tokens + 1, :]
        img_patches = self.emb_dropout(img_patches)

        result = self.Transformer_Encoder(img_patches)
        logits = self.mlp_head(result[:, 0, :])
        logits = nn.Sigmoid()(logits)
        return logits
    
    
