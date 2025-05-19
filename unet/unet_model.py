from .unet_parts import *
from .CBAM import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.se_block1 = SELayer(channel=2048)
        # self.se_block = SELayer(channel=1024)
        self.res1 = OutConv(n_channels,64)
        self.res2 = Res(64,128)
        self.res3 = Res(128,256)
        self.res4 = Res(256,512)
        self.res5 = Res(512,1024)
        self.CBAM1 = CBAM(64)
        self.CBAM2 = CBAM(128)
        self.CBAM3 = CBAM(256)
        self.CBAM4 = CBAM(512)
        self.CBAM5 = CBAM(1024)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512) 
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 下采样
        # input_channels = x.size(1)
        # # 动态定义self.inc
        # self.inc = DoubleConv(input_channels, 64)
        # print(x.shape)
        res_1 = x
        x1 = self.inc(x)
        res_1 = self.res1(res_1)
        x1 = torch.add(x1, res_1)
        x1 = F.relu(x1)
        res_2 = x1
        x2 = self.down1(x1)
        # print(x2.shape)
        res_2 = self.res2(res_2)
        # print(res_2.shape)
        x2 = torch.add(x2, res_2)
        x2 = F.relu(x2)
        # x2 = self.CBAM1(x2)
        res_3 = x2
        x3 = self.down2(x2)
        res_3 = self.res3(res_3)
        x3 = torch.add(x3, res_3)
        x3 = F.relu(x3)
        # x3 = self.CBAM2(x3)
        res_4 = x3
        x4 = self.down3(x3)
        res_4 = self.res4(res_4)
        x4 = torch.add(x4, res_4)
        x4 = F.relu(x4)
        # x4 = self.CBAM3(x4)
        res_5 = x4
        x5 = self.down4(x4)
        res_5 = self.res5(res_5)
        x5 = torch.add(x5, res_5)
        x5 = F.relu(x5)
        # x5 = self.CBAM4(x5)
        # x5 = self.se_block(x5)
        x1 = self.CBAM1(x1)
        x2 = self.CBAM2(x2)
        x3 = self.CBAM3(x3)
        x4 = self.CBAM4(x4)
        x5 = self.CBAM5(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits