import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50
from torch.nn import functional as F
import math





################ ENCODER #####################

# Encoder model that uses a CNNBackbone to encode the input image. Define two heads for the encoder:
# - Gets a representation of the wall channel of the input image
# - Gets a representation of the agent channel of the input image
# - Fuse the two representations using a linear layer to get the final representation

class Encoder(nn.Module):
    def __init__(self, n_kernels, repr_dim, config, dropout=0.1):
        super().__init__()
        self.n_kernels = n_kernels
        self.repr_dim = repr_dim
        self.dropout = dropout
        self.config = config
        self.wall_encoder = self._get_backbone(config.encoder_type)
        self.agent_encoder = self._get_backbone(config.encoder_type)
        self.fusion = nn.Linear(repr_dim*2, repr_dim)

    def _get_backbone(self, backbone):
        if backbone == "cnn":
            return CNNBackbone(self.n_kernels, self.repr_dim, self.dropout, self.config.norm_features)
        elif backbone == "vit":
            return ViTBackbone(image_size=65, patch_size=16, 
                               in_channels=1, embed_dim=self.repr_dim, 
                               num_heads=self.config.num_heads, 
                               mlp_dim=self.config.dim_feedforward,
                               num_layers=self.config.num_layers,
                               dropout=self.config.dropout, norm_features=self.config.norm_features)
    
    def forward(self, x):
        wall_repr = self.wall_encoder(x[:, 0].unsqueeze(1))
        agent_repr = self.agent_encoder(x[:, 1].unsqueeze(1))
        x = torch.cat((wall_repr, agent_repr), dim=-1)
        x = self.fusion(x)
        return x


################# ResNet Encoder ####################
class ResNetEncoder(nn.Module):
    def __init__(self, enc_dim):
        super(ResNetEncoder, self).__init__()

        self.enc_dim = enc_dim
        # Modify the first convolution layer to accept 2 channels
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the output layer to output enc_dim features
        self.resnet.fc = nn.Linear(512, enc_dim)
        
    def forward(self, x):
        
        x = self.resnet(x)
        # Normalize the output
        x = F.normalize(x, dim=-1)
        
        return x

################ CNN Backbone ####################

class CNNSelfAttention(nn.Module):
    
    def __init__(self, n_channels, n_heads):
        super(CNNSelfAttention, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.q_conv = nn.Conv2d(in_channels=n_channels, out_channels = n_channels // n_heads, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=n_channels, out_channels = n_channels // n_heads, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=n_channels, out_channels = n_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
      
    def forward(self, x):
        
        m_batchsize, C, width , height = x.size()
        proj_query  = self.q_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.k_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check

        # Scale Attn Weights by Channel Depth (embedding dimension)
        attention = self.softmax(energy / torch.sqrt(torch.tensor(C))) # BX (N) X (N) 
        proj_value = self.v_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 1 x 1 Convolution for Residual whenever downsmapling (stride > 1)
        self.downsample_residual = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        residual = x
        if self.downsample_residual is not None:
            residual = self.downsample_residual(x)
        x = self.ConvBlock(x)
        return (F.relu(x + residual))


class CNNBackbone(nn.Module):

    def __init__(self, n_kernels, repr_dim, dropout=0.1, norm_features=False):
        super().__init__()
        self.n_kernels = n_kernels
        self.repr_dim = repr_dim
        self.dropout = dropout
        self.norm_features = norm_features

        # 2 x 64 x 64 --> n_kernels x 64 x 64
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(1, self.n_kernels, kernel_size=3), 
            nn.BatchNorm2d(self.n_kernels),
            nn.ReLU(inplace=True)
        )        
        
        # n_kernels x 64 x 64 --> n_kernels * 4 x 16 x 16
        self.ResBlock1 = nn.Sequential(
            ResidualLayer(self.n_kernels, self.n_kernels*2, stride=2),
            ResidualLayer(self.n_kernels*2, self.n_kernels*4, stride=2),
        )
        self.SelfAttn1 = CNNSelfAttention(
            n_channels=self.n_kernels*4,
            n_heads=2 # 1 Head per 32 channels
        )
        self.Bn1 = nn.BatchNorm2d(self.n_kernels*4)
        
        # n_kernels * 2 x 16 x 16 --> n_kernels * 16 x 4 x 4
        self.ResBlock2 = nn.Sequential(
            ResidualLayer(self.n_kernels*4, self.n_kernels*4, stride=2),
            ResidualLayer(self.n_kernels*4, self.n_kernels*4, stride=2),
        )
        self.SelfAttn2 = CNNSelfAttention(
            n_channels=self.n_kernels*4,
            n_heads=2 # 1 Head per 32 channels
        )
        self.Bn2 = nn.BatchNorm2d(self.n_kernels*4)

        self.ResBlock3 = nn.Sequential(
            ResidualLayer(self.n_kernels*4, self.n_kernels*8, stride=1),
            ResidualLayer(self.n_kernels*8, self.n_kernels*8, stride=1),
        )

        self.SelfAttn3 = CNNSelfAttention(
            n_channels=self.n_kernels*8,
            n_heads=2 # 1 Head per 32 channels
        )

        self.Bn3 = nn.BatchNorm2d(self.n_kernels*8)

        self.ResBlock4 = nn.Sequential(
            ResidualLayer(self.n_kernels*8, self.n_kernels*16, stride=1),
            ResidualLayer(self.n_kernels*16, self.n_kernels*16, stride=1),
        )

        self.SelfAttn4 = CNNSelfAttention(
            n_channels=self.n_kernels*16,
            n_heads=2 # 1 Head per 32 channels
        )

        self.Bn4 = nn.BatchNorm2d(self.n_kernels*16)

        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_kernels*16*4*4, out_features=self.repr_dim*2, bias=True),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.repr_dim*2, out_features=self.repr_dim, bias=True)
        )
    
    def forward(self, x):
        x = self.ConvLayer1(x) # 64x64 -> 32x32
        x = self.Bn1(self.SelfAttn1(self.ResBlock1(x))) # 32x32 -> 16x16
        x = self.Bn2(self.SelfAttn2(self.ResBlock2(x))) # 16x16 -> 8x8
        x = self.Bn3(self.SelfAttn3(self.ResBlock3(x)))
        x = self.Bn4(self.SelfAttn4(self.ResBlock4(x)))
        x = self.FC1(x)

        # Normalize the output
        if self.norm_features:
            x = F.normalize(x, dim=-1)
        return x # (batch_size, n_kernels * 16)


############### Transformer Encoder ####################

class PatchEmbedding(nn.Module):
    
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
      super().__init__()
      self.image_size = image_size
      self.patch_size = patch_size
      self.in_channels = in_channels
      self.embed_dim = embed_dim
      
      self.conv_proj = nn.Conv2d(
        in_channels=self.in_channels, 
        out_channels=embed_dim, 
        kernel_size=self.patch_size, 
        stride=self.patch_size
      )

    def forward(self, x):
      bs, c, h, w = x.shape
      n_patches = (h * w) // self.patch_size**2

      x = self.conv_proj(x) # (bs, c, h, w) --> (bs, embed_size, n_h, n_w)
      x = x.reshape(bs, self.embed_dim, n_patches) # (bs, embed_size, n_h, n_w) --> (bs, embed_size, n_patches)
      x = x.permute(0, 2, 1) # (bs, embed_size, n_patches) --> (bs, n_patches, embed_size)
      return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
      super().__init__()
      
      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.head_dim = embed_dim // num_heads
      self.dropout = dropout

      # Create copies of input for each q, k, v weights
      self.qkv = nn.Linear(embed_dim, embed_dim * 3) 
      self.q_norm = nn.LayerNorm(self.head_dim)
      self.k_norm = nn.LayerNorm(self.head_dim)

      self.projection = nn.Linear(self.embed_dim, self.embed_dim)
      self.projection_dropout = nn.Dropout(self.dropout)
      


    def forward(self, x):
      bs, n_patches, embed_size = x.shape

      # Copies input embedding 3 times for q, k and v --> (Concat, bs, num_heads, n_patches, head_dim)
      qkv = self.qkv(x).reshape(bs, n_patches, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
      # 3 x copies (bs, num_heads, n_patches, head_dim)
      q, k, v = qkv.unbind(0)

      q, k = self.q_norm(q), self.k_norm(k)

      # Scaled Dot Product Attn (QK^T) / sqrt(d)
      attn = q @ k.transpose(-2, -1) * math.sqrt(self.head_dim)**-1
      attn = F.softmax(attn, dim=-1)

      x = attn @ v
      x = x.transpose(1, 2).reshape(bs, n_patches, embed_size)
      x = self.projection(x)
      x = self.projection_dropout(x)
      return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        
        self.attention = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_dim),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        residual_1 = x
        x = self.attention(x)
        x = self.layer_norm_1(x) + residual_1

        residual_2 = x
        x = self.ffn(x)
        x = self.layer_norm_2(x) + residual_2

        return x  
    
class ViTBackbone(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, 
                 num_heads, mlp_dim, num_layers, dropout=0.1, norm_features=False):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_features = norm_features

        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, self.in_channels, self.embed_dim)
        n_patches = (self.image_size // self.patch_size)**2

        # Learnable Class Token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
      
        # Learnable Position Embeddings 
        self.position_encoding = nn.Parameter(torch.empty(1, n_patches + 1, self.embed_dim))
        nn.init.trunc_normal_(self.position_encoding, std=0.02)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.embed_dim*(n_patches+1), out_features=self.embed_dim*2, bias=True),
            nn.BatchNorm1d(self.embed_dim*2),
            nn.ReLU(),
            nn.Linear(in_features=self.embed_dim*2, out_features=self.embed_dim, bias=True)
        )


    def forward(self, x):

        bs = x.shape[0]
        x = self.patch_embedding(x)

        # Expan class tokens along batch dimension
        class_token = self.class_token.expand(bs, -1, -1)
    

        # Concatenat class token to each embedding
        x = torch.cat((class_token, x), dim=1)

        # Add positional encoding
        x += self.position_encoding


        for block in self.transformer_blocks:
            x = block(x)

        x = self.FC(x)

        if self.norm_features:
            x = F.normalize(x, dim=-1)
    
        return x





if __name__ == "__main__":
    # Test transformer backbone

    model = CNNBackbone(n_kernels=4, repr_dim=64, dropout=0.1, norm_features=True)
    x = torch.randn(32, 1, 65, 65)
    out = model(x)
    print(out.shape)