import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50

class ResNetEncoder(nn.Module):
    def __init__(self, enc_dim):
        super(ResNetEncoder, self).__init__()

        self.enc_dim = enc_dim
        # Modify the first convolution layer to accept 2 channels
        self.resnet = resnet34(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the output layer to output enc_dim features
        self.resnet.fc = nn.Linear(512, enc_dim)
        
    def forward(self, x):
        
        x = self.resnet(x)
        
        return x

if __name__ == "__main__":
    # Test ResNetEncoder
    enc_dim = 256
    encoder = ResNetEncoder(enc_dim)
    obs = torch.randn((64, 17, 2, 65, 65))
    encoded_features = encoder(obs)
    print(encoded_features.size())
    # print(encoded_features)