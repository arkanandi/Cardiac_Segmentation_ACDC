import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl


class conv_block(pl.LightningModule):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(pl.LightningModule):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(pl.LightningModule):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttU_Net2D(pl.LightningModule):
    def __init__(self, drop):
        super(AttU_Net2D, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout2d(p=drop)
        self.upsample2d = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)

        self.Conv1 = conv_block(ch_in=1, ch_out=48)
        self.Conv2 = conv_block(ch_in=48, ch_out=96)
        self.Conv3 = conv_block(ch_in=96, ch_out=192)
        self.Conv4 = conv_block(ch_in=192, ch_out=384)
        self.Conv5 = conv_block(ch_in=384, ch_out=768)

        self.Up5 = up_conv(ch_in=768, ch_out=384)
        self.Att5 = Attention_block(F_g=384, F_l=384, F_int=192)
        self.Up_conv5 = conv_block(ch_in=768, ch_out=384)

        self.Up4 = up_conv(ch_in=384, ch_out=192)
        self.Att4 = Attention_block(F_g=192, F_l=192, F_int=96)
        self.Up_conv4 = conv_block(ch_in=384, ch_out=192)

        self.Up3 = up_conv(ch_in=192, ch_out=96)
        self.Att3 = Attention_block(F_g=96, F_l=96, F_int=48)
        self.Up_conv3 = conv_block(ch_in=192, ch_out=96)

        self.Up2 = up_conv(ch_in=96, ch_out=48)
        self.Att2 = Attention_block(F_g=48, F_l=48, F_int=24)
        self.Up_conv2 = conv_block(ch_in=96, ch_out=48)

        self.Conv_1x1 = nn.Conv2d(48, 4, kernel_size=(1, 1))

        # deep supervision 2nd decoder block
        self.deep1 = nn.Conv2d(192, 4, kernel_size=(1, 1), padding='same')
        # deep supervision 3rd decoder block
        self.deep2 = nn.Conv2d(96, 4, kernel_size=(1, 1), padding='same')

        self.neg_slope = 1e-2
        self.apply(self.InitWeights_He)

    def InitWeights_He(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.drop(x2)  # dropout

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.drop(x3)  # dropout

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.drop(x4)  # dropout

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.drop(d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.drop(d4)
        d4 = self.Up_conv4(d4)
        ds2 = copy.copy(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.drop(d3)
        d3 = self.Up_conv3(d3)
        ds3_2 = copy.copy(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # Deep supervision
        ds2_1x1_conv = self.deep1(ds2)
        ds1_ds2_sum_upscale = self.upsample2d(ds2_1x1_conv)
        ds3_1x1_conv = self.deep2(ds3_2)
        ds1_ds2_sum_upscale_ds3_sum = torch.add(ds1_ds2_sum_upscale, ds3_1x1_conv)
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample2d(ds1_ds2_sum_upscale_ds3_sum)
        out = torch.add(d1, ds1_ds2_sum_upscale_ds3_sum_upscale)

        return out


# if __name__ == "__main__":
#     model = AttU_Net2D(0.0).cuda()
#     inp = torch.rand(6, 1, 240, 240).cuda()
#     output = model(inp)
#     print("output", output.shape, "Number of parameters", sum(p.numel() for p in model.parameters()))
