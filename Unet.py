import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, num_classes=5):
        super(UNet, self).__init__()

        self.encoder = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = nn.Sequential(*list(self.encoder.children())[0:4])
        self.conv2 = nn.Sequential(*list(self.encoder.children())[5])
        self.conv3 = nn.Sequential(*list(self.encoder.children())[6])
        self.conv4 = nn.Sequential(*list(self.encoder.children())[7])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mid = conv_block(512, 512)
        self.mid_up = nn.Upsample(scale_factor=2, mode='nearest')

        self.up4 = nn.Sequential(
            conv_block(512+512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.up3 = nn.Sequential(
            conv_block(512+256, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.up2 = nn.Sequential(
            conv_block(256+128, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.up1 = nn.Sequential(
            conv_block(128+64, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.up0 = nn.Sequential(
            conv_block(64, 32),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.out = nn.Conv2d(32, num_classes, kernel_size=1)
        self.logits = nn.Softmax(dim=1)
        self.up_conv1 = conv_block(512, 256)
        
        self.decoder = nn.Sequential(
            conv_block(512, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_block(256, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_block(128, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv_block(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # print(max(x))
        x1 = self.conv1(x)
        # print("X1:", x1.shape)
        x2 = self.conv2(x1)
        # print("X2:", x2.shape)
        x3 = self.conv3(x2)
        # print("X3:", x3.shape)
        x4 = self.conv4(x3)
        # print("X4:", x4.shape)
        x5 = self.maxpool(x4)
        # print("X5:", x5.shape)

        x = self.mid(x5)
        x = self.mid_up(x)
        # print("MID:", x.shape)

        x = torch.cat([x, x4], dim=1)
        # print("CAT:", x.shape)
        x = self.up4(x)
        x = torch.cat([x, x3], dim=1)
        # print("UP4:", x.shape)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        # print("UP1:", x.shape)
        x = self.up0(x)

        x = self.out(x)
        # x = self.logits(x)

        return x1
    
if __name__ == '__main__':
    model = UNet().cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y = model(x)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y, torch.randn(1, 64, 64, 64).cuda())
    print(y.shape)