import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from tqdm.notebook import tqdm
from ROPE import RotaryEmbedding

debug = False
torch._dynamo.config.cache_size_limit = 64
#torch._dynamo.config.guard_nn_modules = True

#make alphas for B, L sequences
def make_non_uniform_alphas1D(x, smoothing_ratio=0, return_probability_dist=True):
    assert len(x.shape) == 2
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L = x.shape
    pool_width = int(L*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, L), device=device)
    dimensions = L
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, L, mode="linear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier
    
    base = base.reshape(B, L)
    base = F.avg_pool1d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=1).unsqueeze(1)
    base = base / base.std(axis=1).unsqueeze(1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

#make alphas for B, C, H, W, images
def make_non_uniform_alphas2D(x, smoothing_ratio=0, return_probability_dist=True):
    assert len(x.shape) == 4
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    B, C, H, W = x.shape
    pool_width = int(H*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, H, W), device=device)
    dimensions = H
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions), int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, (int(H), int(H)), mode="bilinear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier

    base = base.reshape(B, H, W)
    base = F.avg_pool2d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=[1,2]).reshape(-1,1,1)
    base = base / base.std(axis=[1,2]).reshape(-1,1,1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

#make alphas for B, C, H, W, D volumes
def make_non_uniform_alphas3D(x, smoothing_ratio=0, return_probability_dist=True):
    assert len(x.shape) == 5
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W, D = x.shape
    pool_width = int(H*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, H, W, D), device=device)
    dimensions = H
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions), int(dimensions), int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, (int(H), int(H), int(H)), mode="trilinear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier

    base = base.reshape(B, H, W, D)
    base = F.avg_pool3d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=[1,2,3]).reshape(-1,1,1,1)
    base = base / base.std(axis=[1,2,3]).reshape(-1,1,1,1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

def scramble_order_tabular(tab_data, return_index=False):
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_type_tab, x_tab, pos_tab1 = tab_data
    x_tab, x_type_tab, pos_tab1, = x_tab.to(device), x_type_tab.to(device), pos_tab1.to(device),
    
    alphas = torch.empty((x_tab.shape[0], x_tab.shape[1]), device=device).uniform_(0, 0.000001)
    
    x = torch.cat([x_tab], axis=1)
    x_type = torch.cat([x_type_tab], axis=1)
    pos1 = torch.cat([pos_tab1], axis=1)
    
    _, index = alphas.sort(axis=1, descending=False)

    seq_order1 = torch.gather(pos1, 1, index)
    x_reordered = torch.gather(x, 1, index)
    x_type_reordered = torch.gather(x_type, 1, index)
    
    if return_index:
        return index, x_type_reordered, x_reordered,  (seq_order1,)
    else:
        return x_type_reordered, x_reordered,  (seq_order1,)

def scramble_order_seq(seq_data, biased_scramble=True, smoothing_ratio=0, return_index=False, simple_scramble=False, alternate_direction=False):
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    x_type, x, pos1, = seq_data
    x, x_type, pos1, = x.to(device), x_type.to(device), pos1.to(device),
    
    alphas = torch.empty((x.shape[0], x.shape[1]), device=device).uniform_(0, 0.000001)
    
    B, L = x.shape
    if biased_scramble:
        if simple_scramble:
            alphas_seq = torch.arange(L, device=device)
            alphas_seq = alphas_seq/alphas_seq.max()
            alphas_seq = alphas_seq.repeat(B, 1).reshape(B,-1)
        else:
            alphas_seq = make_non_uniform_alphas1D(x, smoothing_ratio=smoothing_ratio)
        
        alphas += alphas_seq
    if alternate_direction:
        direction = torch.randint(0,2,(B,1), device=device) * 2 - 1
        alphas *= direction
    
    x = x.reshape(B,-1)
    x_type = x_type.reshape(B,-1)
    pos1 = pos1.reshape(B,-1)
    
    _, index = alphas.sort(axis=1, descending=False)

    seq_order1 = torch.gather(pos1, 1, index)
    x_reordered = torch.gather(x, 1, index)
    x_type_reordered = torch.gather(x_type, 1, index)
    
    if return_index:
        return index, x_type_reordered, x_reordered,  (seq_order1,)
    else:
        return x_type_reordered, x_reordered,  (seq_order1,)

def scramble_order(tab_data, seq_data, biased_scramble=True, smoothing_ratio=0, return_index=False, simple_scramble=False):
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_type_tab, x_tab, pos_tab1 = tab_data
    x_tab, x_type_tab, pos_tab1, = x_tab.to(device), x_type_tab.to(device), pos_tab1.to(device),
    
    x_type, x, pos1, = seq_data
    x, x_type, pos1, = x.to(device), x_type.to(device), pos1.to(device),
    
    alphas = torch.empty((x.shape[0], x_tab.shape[1]+x.shape[1]), device=device).uniform_(0, 0.000001)
    
    B, L = x.shape
    if biased_scramble:
        if simple_scramble:
            alphas_tab = torch.randint(0,2, (x_tab.shape[0],1), device=device).float()
        else:
            alphas_tab = torch.empty((x_tab.shape[0],1), device=device).uniform_(0,1)
        alphas[:,0:x_tab.shape[1]] += alphas_tab
        if simple_scramble:
            alphas_seq = torch.arange(L, device=device)
            alphas_seq = alphas_seq/alphas_seq.max()
            alphas_seq = alphas_seq.repeat(B, 1).reshape(B,-1)
        else:
            alphas_seq = make_non_uniform_alphas1D(x, smoothing_ratio=smoothing_ratio)
        
        alphas[:,x_tab.shape[1]:x_tab.shape[1]+x.shape[1]] += alphas_seq
    
    x = torch.cat([x_tab, x.reshape(B,-1)], axis=1)
    x_type = torch.cat([x_type_tab, x_type.reshape(B,-1)], axis=1)
    pos1 = torch.cat([pos_tab1, pos1.reshape(B,-1), ], axis=1)
    
    _, index = alphas.sort(axis=1, descending=False)

    seq_order1 = torch.gather(pos1, 1, index)
    x_reordered = torch.gather(x, 1, index)
    x_type_reordered = torch.gather(x_type, 1, index)
    
    if return_index:
        return index, x_type_reordered, x_reordered,  (seq_order1,)
    else:
        return x_type_reordered, x_reordered,  (seq_order1,)


#Scrambles the order of input elements to force the model to learn random order autoregressive prediction
#Creates random values called alphas in a tensor of the same shape of the input and sorts the sequence based on them
#Different input types can generate alphas using different strategies
#We use a random strategy for the tabular data and a pseudo random strategy for the image data
def scramble_order2D(tab_data, seq_data, biased_scramble=True, smoothing_ratio=0, return_index=False, simple_scramble=False):
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_type_tab, x_tab, pos_tab1, pos_tab2 = tab_data
    x_tab, x_type_tab, pos_tab1, pos_tab2 = x_tab.to(device), x_type_tab.to(device), pos_tab1.to(device), pos_tab2.to(device)
    x_type, x, pos1, pos2  = seq_data
    x, x_type, pos1, pos2 = x.to(device), x_type.to(device), pos1.to(device), pos2.to(device)

    #
    alphas = torch.empty((x.shape[0], x_tab.shape[1]+(x.shape[2]*x.shape[3])), device=device).uniform_(0, 0.000001)

    B, C, H, W = x.shape
    if biased_scramble:
        alphas_tab = torch.empty((x_tab.shape[0],1), device=device).uniform_(0,1)
        alphas[:,0:x_tab.shape[1]] += alphas_tab

        if simple_scramble:
            alphas_seq = (torch.arange(x.shape[1], device=device).repeat(x.shape[1],1) + torch.arange(x.shape[1], device=device).repeat(x.shape[1],1).T)
            alphas_seq = alphas_seq/alphas_seq.max()
            alphas_seq = alphas_seq.repeat(x.shape[0], 1,1).reshape(B,-1)
        else:
            alphas_seq = make_non_uniform_alphas2D(x, smoothing_ratio=smoothing_ratio).reshape(B,-1)
        alphas[:,x_tab.shape[1]:] += alphas_seq
    
    x = torch.cat([x_tab, x.reshape(B,-1)], axis=1)
    x_type = torch.cat([x_type_tab, x_type.reshape(B,-1)], axis=1)
    pos1 = torch.cat([pos_tab1, pos1.reshape(B,-1)], axis=1)
    pos2 = torch.cat([pos_tab2, pos2.reshape(B,-1)], axis=1)
        
    _, index = alphas.sort(axis=1, descending=False)

    seq_order1 = torch.gather(pos1, 1, index)
    seq_order2 = torch.gather(pos2, 1, index)
    x_reordered = torch.gather(x, 1, index)
    x_type_reordered = torch.gather(x_type, 1, index)
    
    if return_index:
        return index, x_type_reordered, x_reordered,  (seq_order1, seq_order2)
    else:
        return x_type_reordered, x_reordered,  (seq_order1, seq_order2)


def make_diag(length):
    array = torch.ones((length+1, length))
    for i in range(0,length+1):
        array[i,i:] = 0
    return array

class ToBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() # pass-through

class BinaryQuantizer(nn.Module):
    def forward(self, x, add_noise=False, round_during_training=True):
        if add_noise and self.training:
            x = x + torch.empty_like(x).normal_()
        x = torch.sigmoid(x)
        if not self.training or round_during_training: 
            x = ToBinary.apply(x)
        return x

class ResBlockCNN(nn.Module):
    def __init__(self, n_blocks, n_channels, in_channels = None, out_channels = None):
        super(ResBlockCNN, self).__init__()
        
        self.translation_layer_in = None
        if in_channels is not None:
            self.translation_layer_in = nn.Sequential(         
                nn.Conv2d(
                    in_channels=in_channels,              
                    out_channels=n_channels,            
                    kernel_size=3,              
                    stride=1,                   
                    padding="same",                  
                ))
        
        self.translation_layer_out = None
        if out_channels is not None:
            self.translation_layer_out = nn.Sequential(         
                nn.GroupNorm(1, n_channels),
                nn.Mish(),
                nn.Conv2d(
                    in_channels=n_channels,              
                    out_channels=out_channels,            
                    kernel_size=3,              
                    stride=1,                   
                    padding="same",                  
                ))
        
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(         
                nn.GroupNorm(1, n_channels),
                nn.Mish(),
                nn.Conv2d(
                    in_channels=n_channels,              
                    out_channels=n_channels,            
                    kernel_size=3,              
                    stride=1,                   
                    padding="same"),
                nn.Mish(),
                nn.Conv2d(
                    in_channels=n_channels,              
                    out_channels=n_channels,            
                    kernel_size=3,              
                    stride=1,                   
                    padding="same")))                        

            self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        if self.translation_layer_in is not None:
            x = self.translation_layer_in(x)
            
        for block in self.blocks:
            x = block(x) + x
            
        if self.translation_layer_out is not None:
            x = self.translation_layer_out(x)
            
        return x

class VAEEncoder(nn.Module):
    def __init__(self, d1=64, d2=128,translation_layer=2048, n_bits=8, n_blocks=4):
        super(VAEEncoder, self).__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(
                    in_channels=3,              
                    out_channels=d1,            
                    kernel_size=3,              
                    stride=1,                   
                    padding=1),
            nn.Mish(),
            nn.GroupNorm(1, d1),
            nn.Conv2d(
                in_channels=d1,              
                out_channels=d2,            
                kernel_size=4,              
                stride=2,                   
                padding=1,                  
            ),                              
            nn.Mish())#128
        
        self.enc2 = nn.Sequential(
            nn.GroupNorm(1, d2),
            nn.Conv2d(
                in_channels=d2,              
                out_channels=translation_layer,            
                kernel_size=4,              
                stride=2,                   
                padding=1,                  
            ),                              
            nn.Mish())#64

        self.enc_to_mid = nn.Sequential(
            ResBlockCNN(
                n_blocks=n_blocks,
                in_channels=translation_layer,
                n_channels=translation_layer,
                out_channels=n_bits))

        self.binary_quantize = BinaryQuantizer() 
    
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc_to_mid(x)
        x = self.binary_quantize(x)
        return x

class VAEDecoder(nn.Module):
    def __init__(self, d1=64, d2=128, translation_layer=2048, n_bits=8, n_blocks=4):
        super(VAEDecoder, self).__init__()

        self.mid_to_dec = nn.Sequential(
            ResBlockCNN(
                n_blocks=n_blocks,
                in_channels=n_bits,
                n_channels=translation_layer,
                out_channels=translation_layer))
        
        self.dec2 = nn.Sequential(         
            nn.ConvTranspose2d(
                in_channels=translation_layer,              
                out_channels=d2,            
                kernel_size=4,              
                stride=2,                   
                padding=1,
                dilation=1
            ),
            nn.Mish(),
            nn.GroupNorm(1, d2))
        
        self.dec1 = nn.Sequential(         
            nn.ConvTranspose2d(
                in_channels=d2,              
                out_channels=d1,            
                kernel_size=4,              
                stride=2,                   
                padding=1,
                dilation=1
            ),
            nn.Mish(),
            nn.GroupNorm(1, d1),
            nn.Conv2d(
                    in_channels=d1,              
                    out_channels=3,            
                    kernel_size=3,              
                    stride=1,                   
                    padding=1))
        
    
    def forward(self, x):
        x = self.mid_to_dec(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return x

class OBQ_VAE(nn.Module):
    def __init__(self, d1=32, d2=64, n_blocks=4, 
                 translation_layer=2048, n_bits=256):
        super(OBQ_VAE, self).__init__()
        self.n_bits = n_bits
        self.encoder = VAEEncoder(d1=d1, d2=d2, translation_layer=translation_layer, n_blocks=n_blocks, n_bits=n_bits)
        
        self.decoder = VAEDecoder(d1=d1, d2=d2, translation_layer=translation_layer, n_blocks=n_blocks, n_bits=n_bits)
    
    def forward(self, x, max_features=None, num_allowed_nodes=None, new_mid = None):

        mid = self.encoder(x)
        
        if new_mid is not None:
            mid = new_mid

        out = self.decoder(mid)
        
        return mid, out


####################################################################################
# Transformer Code Autoencoder
####################################################################################

#Applies a binary mask to a vector (the embedding) then does things to allow the network to gracefully handle masked items and distinguish them from actual 0s
#For an N length embedding vector:
#1) Multiply by the binary mask, zeroing out masked item
#2) Concat the mask and embedding vector to make a 2 X N matrix
#3) Multiply each element in this matrix by a learned weight and add a learned bias term
#4) Pass each Mask/Embedding element pair through a neural network (same network for all items, applied separately)
class ApplyMask(nn.Module):
    def __init__(self, num_bits, is_binary):
        super(ApplyMask, self).__init__()
        self.is_binary = is_binary
        self.element_weight = nn.Parameter(torch.ones((num_bits, 2)) + torch.normal(0,0.01,(num_bits, 2)))
        self.element_bias = nn.Parameter(torch.zeros((num_bits, 2)) + torch.normal(0,0.01,(num_bits, 2)))
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(512, 1)
        self.swiglu = SwiGLU()

    def forward(self, x, mask):
        in_shape = x.shape
        if self.is_binary:
            x = x*2 - 1
        x = x * mask
        x = torch.stack([x,mask], 2)
        x = x * self.element_weight + self.element_bias
        return self.fc2(self.swiglu(self.fc1(x))).reshape(in_shape)

class ProbDropout(nn.Module):
    def __init__(self, min_p=0, max_p=1):
        super(ProbDropout, self).__init__()
        self.min_p = min_p
        self.max_p = max_p
        
    def forward(self, x, dropout_mask=None):
        if self.training:
            if dropout_mask is None:
                mask = torch.empty_like(x).uniform_(0,1) > torch.empty((x.shape[0],1,1), device=x.device).uniform_(self.min_p, self.max_p)
            else:
                mask = dropout_mask
            x = x * mask
        return x

#Implements Swish gated linear units
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

#Implements the positionwise feedforward network for the transformer
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff*2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU()

    def forward(self, x):
        return self.fc2(self.swiglu(self.fc1(x)))

#Implements a basic feedforward network applied to translate transformer outputs into the enbedding (encoder) and from the embedding back to the input to a transformer (decoder)
class TranslationFF(nn.Module):
    def __init__(self, d_reserved, d_model, d_ff):
        super(TranslationFF, self).__init__()
        self.norm = nn.LayerNorm(d_reserved+d_model)
        self.fc1 = nn.Linear(d_reserved+d_model, d_ff*2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU()
        
    
    def forward(self, reserved, x):
        x_base = x.clone()
        x = torch.cat([reserved, x], axis=1)
        return self.fc2(self.swiglu(self.fc1(self.norm(x)))) + x_base

#Implements a layer of the transformer encoder with rotary positional encodings
#Can rotate keys/queries across multiple axis if input is more than 1d (like an image)
#A portion of each k/v is left unrotated, how much depends on how many axis need to be encoded
class EncoderLayer(nn.Module):
    def __init__(self, n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=1, dropout_max_p=0):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_ff = d_ff
        self.num_rotation_axis = num_rotation_axis
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.type_embedding = nn.Embedding(n_type_embedding, 2 * d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dim_per_rope = (d_model//num_heads)//(num_rotation_axis+1)
        self.rotary_emb = RotaryEmbedding(dim = self.dim_per_rope, cache_if_possible=False)
        self.dropout = ProbDropout(max_p=dropout_max_p)
        

    #takes type, value, position, triplets
    #each is a vector of the same length where each element describes type, value, or position about the input data
    #seq_order is a tuple which can contain multiple axis
    def forward(self, x_type, x_value, seq_order):
        B, T, C = x_value.size()

        #eake queries, keys, values
        q, k, v  = self.c_attn(self.norm1(x_value)).split(self.d_model, dim=2)

        #encodes the TYPE of value into the key/queries by adding a linear projection to each k/q
        q_embed, k_embed = self.type_embedding(x_type).split(self.d_model, dim=2)
        q = q + q_embed
        k = k + k_embed

        #break k/q/v into distinct heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #Add rotations to the correct portion of the keys/queries using the positions listed in the seq_order tuple
        for i, s in enumerate(seq_order):
            q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s.unsqueeze(1))
            k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s.unsqueeze(1))

        #Use flash attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.dropout(attn_output)
        
        x_value = x_value + attn_output
        
        ff_output = self.feed_forward(self.norm2(x_value))
        ff_output = self.dropout(ff_output)
        
        x_value = x_value + ff_output
        return x_value

#Implements a tranformer encoder which produces a single output vector
#Options allow it to encode into binary vectors and to create embeddings based on partial inputs
class Encoder(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=False, learn_partial_encodings=False, num_rotation_axis=1, dropout_max_p=0):
        super(Encoder, self).__init__()
        self.binary_encodings = binary_encodings
        self.n_bits = n_bits
        self.learn_partial_encodings = learn_partial_encodings
        self.embedding = nn.Embedding(n_embedding, d_model)
        self.type_embedding = nn.Embedding(n_type_embedding, d_model)
        encoder = []
        for i in range(depth):
            encoder.append(EncoderLayer(n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=num_rotation_axis, dropout_max_p=dropout_max_p))
        self.encoder = nn.ModuleList(encoder)
        translation_layers = []
        for _ in range(n_translation_layers):
            translation_layers.append(TranslationFF(d_model, d_model, d_ff))
        self.translation = nn.ModuleList(translation_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.lin_out = nn.Linear(d_model, self.n_bits)
        self.binary_quantize = BinaryQuantizer()
        
    def forward(self, x_type, x_value, seq_order):
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, L = x_type.shape
        mask = torch.ones(L, device=device).repeat(B,1)
        if self.training and self.learn_partial_encodings:
            mask = torch.arange(L, device=device).repeat(B,1)/L < torch.empty((B,1), device=device).uniform_(0,1)
        
        #add pooling token pos
        new_seq_order = []
        for s in seq_order:
            new_seq_order.append(torch.cat([torch.zeros((B,1), device=device),s*mask], axis=1).float())
        seq_order = new_seq_order

        #add pooling token, pooling is done at this token. If strings include CLS the CLS will be scrambled like all other tokens and just marks the start
        x_value = torch.cat([torch.ones((B,1), device=device),x_value*mask], axis=1).long()
        x_type = torch.cat([torch.ones((B,1), device=device),x_type*mask], axis=1).long()

        #initial input to the transformer
        x_value = self.embedding(x_value) + self.type_embedding(x_type)
        
        for block in self.encoder:
            x_value = block(x_type, x_value, seq_order=seq_order)
        x_value = x_value[:,0,:]
        base_value = x_value.clone()
        for block in self.translation:
            x_value = block(base_value, x_value)
        x_value = self.lin_out(self.norm_out(x_value))

        if self.binary_encodings:
            x_value = self.binary_quantize(x_value)
        elif self.training: 
            x_value = x_value + torch.empty_like(x_value).normal_(std=0.01)
            
        return x_value

#Implements a layer of the transformer decoder with rotary positional encodings
#Can rotate keys/queries across multiple axis if input is more than 1d (like an image)
#A portion of each k/v is left unrotated, how much depends on how many axis need to be encoded
class DecoderLayer(nn.Module):
    def __init__(self, n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=1, dropout_min_p=0, dropout_max_p=0):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_ff = d_ff
        self.num_rotation_axis = num_rotation_axis
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.type_embedding = nn.Embedding(n_type_embedding, 2 * d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dim_per_rope = (d_model//num_heads)//(num_rotation_axis+1)
        self.rotary_emb = RotaryEmbedding(dim = self.dim_per_rope, cache_if_possible=False)
        self.dropout = ProbDropout(min_p=dropout_min_p, max_p=dropout_max_p)
    
    #CURRENT position/type is encoded into the queries while "NEXT" position is encoded into the keys
    #This allows information to be routed to allow for autoregressive prediction of "NEXT" tokens
    #Predicting "NEXT" tokens of unknown position and type is impossible so we have to give the self attenton mechanism this information
    def forward(self, x_type, x_value, seq_order, dropout_mask=None):
        B, T, C = x_value.size()
        q, k, v  = self.c_attn(self.norm1(x_value)).split(self.d_model, dim=2)
        q_embed, k_embed = self.type_embedding(x_type).split(self.d_model, dim=2)

        #Encodings of type are offset for queries/keys in the decoder. See above note.
        q = q + q_embed[:,:-1]
        k = k + k_embed[:,1:]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        
        #Encodings of position are offset for queries/keys in the decoder. See above note.
        for i, s in enumerate(seq_order):
            q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s[:,:-1].unsqueeze(1))
            k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s[:,1:].unsqueeze(1))

        #Flash attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.dropout(attn_output)
        
        x_value = x_value + attn_output
        ff_output = self.feed_forward(self.norm2(x_value))
        ff_output = self.dropout(ff_output, dropout_mask=dropout_mask)
        x_value = x_value + ff_output
        return x_value

#Implements a tranformer decoder which takes a single vector embedding and autoregressively decodes it in a specified order
#Options allow it to use binary vectors and to create hierarchical embeddings by masking parts of the input embeddings
class Decoder(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=False, ordered_encodings=False, num_rotation_axis=1, dropout_min_p=0, dropout_max_p=0):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.ordered_encodings = ordered_encodings
        self.binary_encodings = binary_encodings
        self.n_bits = n_bits
        self.dropout_min_p = dropout_min_p
        self.dropout_max_p = dropout_max_p
        self.apply_mask = ApplyMask(n_bits, is_binary=binary_encodings)
        self.embedding = nn.Embedding(n_embedding, d_model)
        self.type_embedding = nn.Embedding(n_type_embedding, d_model)
        decoder = []
        for i in range(depth):
            decoder.append(DecoderLayer(n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=num_rotation_axis, dropout_min_p=dropout_min_p, dropout_max_p=dropout_max_p))
        self.decoder = nn.ModuleList(decoder)
        self.lin_in = nn.Linear(self.n_bits, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.lin_out = nn.Linear(d_model, n_embedding)
        
        translation_layers = []
        for _ in range(n_translation_layers):
            translation_layers.append(TranslationFF(d_model,d_model, d_ff))
        self.translation = nn.ModuleList(translation_layers)
        self.register_buffer('mask', make_diag(self.n_bits))

    #makes a mask used to force the encoder to push more information into earlier dimensions of the embedding vector
    #masks the last X items in the vector where X is random
    #this forces more information into earlier dimensions
    def make_rand_mask(self, x_in, num_allowed_nodes=None):
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rand_mask = None
        B = x_in.shape[0]
        if self.ordered_encodings:
            #mask nodes to force more information into fewer nodes
            if num_allowed_nodes is not None:
                rand_mask = self.mask[num_allowed_nodes].repeat(B,1)
            else:
                r = torch.empty((B,), device=device).uniform_(0,1) ** 2
                r = r - r.min()
                r = r / r.max()
                r = torch.nan_to_num(r)
                rand_index = (r * (self.mask.shape[0]-1)).round().int()
                if self.training:
                    rand_mask = self.mask[rand_index]
                else:
                    rand_mask = self.mask[-1].repeat(B,1)
        else:
            rand_mask = self.mask[-1].repeat(B,1)
        return rand_mask

    def forward(self, x_type, x_value, seq_order, enc=None, num_allowed_nodes=None):
        if debug:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #makes the mask which is randomly applied to the input embedding to force a hierarchy
        #masks the last X items in the vector where X is random
        #this forces more information into earlier dimensions
        if enc is None:
            enc = torch.zeros((x_value.shape[0],self.n_bits), device=device)
            enc_mask = torch.zeros(enc.shape, device=device)
        else:
            enc_mask = self.make_rand_mask(x_value, num_allowed_nodes=num_allowed_nodes)

        #add a leading zero to the vectors containing the positions of each element due to the right shift of decoder inputs needed for autoregressive causal masked training
        new_seq_order = []
        for s in seq_order:
            new_seq_order.append(torch.cat([torch.zeros((x_value.shape[0],1), device=device),s], axis=1).float())
        seq_order = new_seq_order

        #apply translate from embedding to a middle form before giving the vector to the transformer decoder
        enc = self.apply_mask(enc, enc_mask)
        enc = self.norm_in(self.lin_in(enc))
        base_value = enc.clone()
        
        for block in self.translation:
            enc = block(base_value, enc)
        
        enc = enc.unsqueeze(1)
        
        seq_len = x_value.shape[1]

        #add a leading one to the vectors containing the positions of each element due to the right shift of decoder inputs needed for autoregressive causal masked training
        #the token "1" identifies that this is the input token
        x_type = torch.cat([torch.ones((x_type.shape[0],1), device=device),x_type], axis=1).long()
        x_value = self.embedding(x_value) + self.type_embedding(x_type[:,:-1])

        #make single dropout mask for whole decoder, used for every layer
        dropout_mask = torch.empty_like(x_value).uniform_(0,1) > torch.empty((x_value.shape[0],1,1), device=x_value.device).uniform_(self.dropout_min_p, self.dropout_max_p) 
        #dropout_mask = None

        #add the embedding to the input at the first position. Shift the sequence right one element for autoregressive training
        x_value = torch.cat([enc, x_value], axis=1)[:,0:seq_len,:]
        for block in self.decoder:
            x_value = block(x_type, x_value, seq_order=seq_order, dropout_mask=dropout_mask)

        x_value = self.lin_out(self.norm_out(x_value)).permute(0,2,1)
        return x_value

#Base Random Order Autoregressive Transformer Autoencoder class
#n_embedding = number of distinct VALUES for inputs
#n_type_embedding = number of distinct TYPES of inputs
#d_model base model dimension
#num_heads = number of attention heads, heads are of size d_modes//num_heads
#d_ff = dimensionality of the feedforward dimension of the feedforward layer, should be larger than d_model 2X or 4X as large are common
#depth = number of transformer layers in the encoder and decoder respectively. i.e. 8 = 8 encoder layers + 8 decoder layers
#n_translation_layers = number of feedforward layers after the encoder to translate output -> embedding and before the decoder to translate embedding -> decoder input
#n_bits = dimensionality of the embedding
#binary_encodings = round embedding values to binary. T/F
#ordered_encodings = learn hierarchical embedding vector
#make_encodings = learn to make embeddings T/F, this model works fine as a decoder only model, it doesn't have to make embeddings
#learn_partial_encodings = learn to make embeddings from incomplete input
#num_rotation_axis = number of positional embedding axis needed for the input. i.e tabular data = 0, sequence = 1, image = 2, etc.
class DisorderTransformer(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits=256, binary_encodings=False, ordered_encodings=False, 
                 make_encodings=True, learn_partial_encodings=False, num_rotation_axis=1, dropout_min_p=0, dropout_max_p=0):
        super(DisorderTransformer, self).__init__()
        self.n_bits = n_bits
        self.make_encodings = make_encodings
        self.ordered_encodings = ordered_encodings
        self.binary_encodings = binary_encodings

        if make_encodings:
            self.encoder = Encoder(n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=binary_encodings, learn_partial_encodings=learn_partial_encodings, num_rotation_axis=num_rotation_axis)
        
        self.decoder = Decoder(n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=binary_encodings, ordered_encodings=ordered_encodings, num_rotation_axis=num_rotation_axis, dropout_min_p=dropout_min_p, dropout_max_p=dropout_max_p)
        
        self.register_buffer('mask', make_diag(self.n_bits))
        self.best_complexity = -1
        
    def forward(self, decoder_input, encoder_input=None, enc=None, num_allowed_nodes=None):
        if encoder_input is None:
            encoder_input = decoder_input
        encoder_type, encoder_value, encoder_seq_order = encoder_input
        if enc is None and self.make_encodings:
            enc = self.encoder(encoder_type, encoder_value, encoder_seq_order)

        decoder_type, decoder_value, decoder_seq_order = decoder_input
        out = self.decoder(decoder_type, decoder_value, decoder_seq_order, enc, num_allowed_nodes=num_allowed_nodes)
        
        return enc, out