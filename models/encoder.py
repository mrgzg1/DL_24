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
    def __init__(self, repr_dim, config, dropout=0.1):
        super().__init__()
        self.repr_dim = repr_dim
        self.dropout = dropout
        self.config = config
        self.wall_encoder = self._get_backbone(config, "wall",)
        self.agent_encoder = self._get_backbone(config, "agent")
        self.fusion = nn.Linear(repr_dim*2, repr_dim)

    def _get_backbone(self, config, for_who):
        backbone = config.encoder_type
        if backbone == "cnn":
            num_layers = config.wall_n_layers if for_who == "wall" else config.agent_n_layers
            num_kernels = config.wall_n_kernels if for_who == "wall" else config.agent_n_kernels
            return CNNBackbone(num_kernels, self.repr_dim, self.dropout, self.config.norm_features, num_layers=num_layers)
        elif backbone == "vit":
            raise NotImplementedError
    
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

    def __init__(self, n_kernels, repr_dim, dropout=0.1, norm_features=False, num_layers=4):
        super().__init__()
        assert num_layers >= 2  # Need at least 2 layers
        self.n_kernels = n_kernels
        self.repr_dim = repr_dim
        self.dropout = dropout
        self.norm_features = norm_features
        self.num_layers = int(num_layers)  # Ensure integer

        # Initial conv layer: 2 x 64 x 64 --> n_kernels x 64 x 64
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(1, self.n_kernels, kernel_size=3), 
            nn.BatchNorm2d(self.n_kernels),
            nn.ReLU(inplace=True)
        )        
        
        # First two special layers with stride=2
        self.ResBlock1 = nn.Sequential(
            ResidualLayer(self.n_kernels, self.n_kernels*2, stride=2),
            ResidualLayer(self.n_kernels*2, self.n_kernels*4, stride=2),
        )
        self.SelfAttn1 = CNNSelfAttention(
            n_channels=self.n_kernels*4,
            n_heads=2
        )
        self.Bn1 = nn.BatchNorm2d(self.n_kernels*4)
        
        self.ResBlock2 = nn.Sequential(
            ResidualLayer(self.n_kernels*4, self.n_kernels*4, stride=2),
            ResidualLayer(self.n_kernels*4, self.n_kernels*4, stride=2),
        )
        self.SelfAttn2 = CNNSelfAttention(
            n_channels=self.n_kernels*4,
            n_heads=2
        )
        self.Bn2 = nn.BatchNorm2d(self.n_kernels*4)

        # Additional layers with stride=1
        self.additional_layers = nn.ModuleList()
        curr_channels = self.n_kernels*4
        for i in range(self.num_layers - 2):  # -2 because we already have 2 special layers
            # Cap channel growth after a certain point to prevent memory explosion
            if i < 3:  # First 3 additional layers can grow channels
                next_channels = self.n_kernels * 8 if i == 0 else self.n_kernels * 16
            else:  # After that, maintain constant channel count
                next_channels = curr_channels
                
            self.additional_layers.append(
                nn.ModuleDict({
                    'resblock': nn.Sequential(
                        ResidualLayer(curr_channels, next_channels, stride=1),
                        ResidualLayer(next_channels, next_channels, stride=1),
                    ),
                    'attention': CNNSelfAttention(
                        n_channels=next_channels,
                        n_heads=2
                    ),
                    'bn': nn.BatchNorm2d(next_channels)
                })
            )
            curr_channels = next_channels

        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=curr_channels*4*4, out_features=self.repr_dim*2, bias=True),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.repr_dim*2, out_features=self.repr_dim, bias=True)
        )
    
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.Bn1(self.SelfAttn1(self.ResBlock1(x)))
        x = self.Bn2(self.SelfAttn2(self.ResBlock2(x)))
        
        # Process additional layers
        for layer in self.additional_layers:
            x = layer['bn'](layer['attention'](layer['resblock'](x)))

        x = self.FC1(x)

        # Normalize the output
        if self.norm_features:
            x = F.normalize(x, dim=-1)
        return x

if __name__ == "__main__":
    # Test transformer backbone

    model = CNNBackbone(n_kernels=4, repr_dim=64, dropout=0.1, norm_features=True)
    x = torch.randn(32, 1, 65, 65)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    model = CNNBackbone(n_kernels=4, repr_dim=64, dropout=0.1, norm_features=True, num_layers=6)
    x = torch.randn(32, 1, 65, 65)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    model = CNNBackbone(n_kernels=4, repr_dim=64, dropout=0.1, norm_features=True, num_layers=8)
    x = torch.randn(32, 1, 65, 65)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    model = CNNBackbone(n_kernels=4, repr_dim=64, dropout=0.1, norm_features=True, num_layers=10)
    x = torch.randn(32, 1, 65, 65)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")