import torch
from torch import nn
import torch.nn.functional as F
from math import log

# SR3 UNet architecture from https://github.com/aditya-nutakki/SR3
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, n_times):
        super().__init__()

        # Encoder
        self.e1 = EncoderBlock(input_channels, 64, n_times=n_times)
        self.e2 = EncoderBlock(64, 128, n_times=n_times)
        self.e3 = EncoderBlock(128, 256, n_times=n_times)
        self.ea3 = AttentionBlock(256)
        self.e4 = EncoderBlock(256, 512, n_times=n_times)
        self.ea4 = AttentionBlock(512)

        # Bottleneck
        self.b1 = ConvBlock(512, 1024, n_times=n_times) 
        self.ba1 = AttentionBlock(1024)

        # Decoder
        self.d1 = DecoderBlock(1024, 512, n_times=n_times)
        self.da1 = AttentionBlock(512)
        self.d2 = DecoderBlock(512, 256, n_times=n_times)
        self.da2 = AttentionBlock(256)
        self.d3 = DecoderBlock(256, 128, n_times=n_times)
        self.d4 = DecoderBlock(128, 64, n_times=n_times)

        self.outputs = nn.Conv2d(64, output_channels, kernel_size=1, padding=0)

    def forward(self, x, timestep):
        #Encoding
        s1, p1 = self.e1(x, timestep)
        s2, p2 = self.e2(p1, timestep)
        s3, p3 = self.e3(p2, timestep)
        p3 = self.ea3(p3)
        s4, p4 = self.e4(p3, timestep)
        p4 = self.ea4(p4)

        #Bottleneck
        b1 = self.b1(p4, timestep)
        b1 = self.ba1(b1)

        #Decoding
        d1 = self.d1(b1, s4, timestep)
        d1 = self.da1(d1)
        d2 = self.d2(d1, s3, timestep)
        d2 = self.da2(d2)
        d3 = self.d3(d2, s2, timestep)
        d4 = self.d4(d3, s1, timestep)

        outputs = self.outputs(d4)
        return outputs

class GammaEncoding(nn.Module):
    def __init__(self, embedding_dims):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.linear = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.lrelu = nn.LeakyReLU()

    def forward(self, noise_level):
        count = self.embedding_dims // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count

        encoding = noise_level.unsqueeze(1) * torch.exp(log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return self.lrelu(self.linear(encoding))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_times):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.embedding_dims = out_channels
        self.embedding = GammaEncoding(self.embedding_dims)
        self.relu = nn.ReLU()
        
    def forward(self, x, timestep):
        time_embedding = self.embedding(timestep).view(-1, self.embedding_dims, 1, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + time_embedding
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_times):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, n_times = n_times)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x, timestep):
        x = self.conv(x, timestep)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_times):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(2*out_channels, out_channels, n_times = n_times)

    def forward(self, x, skip, timestep):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x, timestep)
        return x
    

class AttentionBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads = 4) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.ln = nn.LayerNorm(embedding_dims)

        self.mhsa = MultiHeadSelfAttention(embedding_dims = embedding_dims, num_heads = num_heads)

        self.ff = nn.Sequential(
            nn.LayerNorm(self.embedding_dims),
            nn.Linear(self.embedding_dims, self.embedding_dims),
            nn.GELU(),
            nn.Linear(self.embedding_dims, self.embedding_dims),
        )

    def forward(self, x):
        bs, c, sz, _ = x.shape
        x = x.view(-1, self.embedding_dims, sz * sz).swapaxes(1, 2)
        x_ln = self.ln(x)
        _, attention_value = self.mhsa(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, c, sz, sz)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims, num_heads = 4) -> None:
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads

        assert self.embedding_dims % self.num_heads == 0, f"{self.embedding_dims} not divisible by {self.num_heads}"
        self.head_dim = self.embedding_dims // self.num_heads

        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)
        self.wv = nn.Linear(self.head_dim, self.head_dim)

        self.wo = nn.Linear(self.embedding_dims, self.embedding_dims)

    def attention(self, q, k, v):
        attn_weights = F.softmax((q @ k.transpose(-1, -2))/self.head_dim**0.5, dim = -1)
        return attn_weights, attn_weights @ v        

    def forward(self, q, k, v):
        bs, img_sz, c = q.shape

        q = q.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        attn_weights, o = self.attention(q, k, v)
        
        o = o.transpose(1, 2).contiguous().view(bs, img_sz, self.embedding_dims)
        o = self.wo(o)

        return attn_weights, o