import torch
from torch import nn
from torch.nn import functional as F
import math





################ ENCODER #####################

# Encoder model that uses a CNNBackbone to encode the input image. Define two heads for the encoder:
# - Gets a representation of the wall channel of the input image
# - Gets a representation of the agent channel of the input image
# - Fuse the two representations using a linear layer to get the final representation

class Encoder(nn.Module):
    def __init__(self, config, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.config = config
        self.wall_encoder = self._get_backbone(config.encoder_type)
        self.agent_encoder = self._get_backbone(config.encoder_type, "agent")
        self.repr_dim = config.repr_dim
        #self.fusion = nn.Linear(config.repr_dim*2, config.repr_dim)
        self.fusion = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(in_features=config.repr_dim*2, out_features=config.repr_dim*4, bias=True),
                        nn.Dropout(p=self.dropout),
                        nn.ReLU(),
                        nn.Linear(in_features=self.repr_dim*4, out_features=self.repr_dim, bias=True))

    def _get_backbone(self, backbone, for_who="all"):
        if backbone == "cnn":
            return CNNBackbone(self.config.num_kernels, self.config.repr_dim, self.dropout, self.config.norm_features)
        elif backbone == "cnn-new":
            if for_who == "agent":
                return CNNBackbone(self.config.num_kernels//2, self.config.repr_dim, self.dropout, self.config.norm_features)
            else:
                return CNNBackbone(self.config.num_kernels, self.config.repr_dim, self.dropout, self.config.norm_features)
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

        #self._init_weights()

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
            ResidualLayer(self.n_kernels*4, self.n_kernels*8, stride=2),
            ResidualLayer(self.n_kernels*8, self.n_kernels*16, stride=2),
        )
        self.SelfAttn2 = CNNSelfAttention(
            n_channels=self.n_kernels*16,
            n_heads=2 # 1 Head per 32 channels
        )
        self.Bn2 = nn.BatchNorm2d(self.n_kernels*16)
    
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
        x = self.FC1(x)

        # Normalize the output
        if self.norm_features:
            x = F.normalize(x, dim=-1)
        return x # (batch_size, n_kernels * 16)


if __name__ == "__main__":
    # Test transformer backbone

    model = CNNBackbone(
        image_size=65, 
        patch_size=16, 
        in_channels=2, 
        embed_dim=128, 
        num_heads=8, 
        mlp_dim=256, 
        num_layers=2
    )

    obs = torch.randn((64, 2, 65, 65))
    pred_enc = model(obs)

    print(pred_enc.size())
