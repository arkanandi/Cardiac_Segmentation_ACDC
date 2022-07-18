import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchsummary import summary


def double_conv3d(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'),
        nn.BatchNorm3d(out_c, eps=1e-05, momentum=0.1),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(out_c, out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same'),
        nn.BatchNorm3d(out_c, eps=1e-05, momentum=0.1),
        nn.LeakyReLU(inplace=True)
    )
    return conv


class Unet_3d(pl.LightningModule):
    def __init__(self, drop):
        super(Unet_3d, self).__init__()

        self.max_pool3d = nn.MaxPool3d(kernel_size=(2, 2, 1), ceil_mode=True)
        self.drop = nn.Dropout3d(p=drop)
        self.upsample3d = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)

        self.down_conv1 = double_conv3d(1, 26)
        self.down_conv2 = double_conv3d(26, 52)
        self.down_conv3 = double_conv3d(52, 104)
        self.down_conv4 = double_conv3d(104, 208)
        self.down_conv5 = double_conv3d(208, 416)

        self.up_conv1 = double_conv3d(624, 208)
        self.up_conv2 = double_conv3d(312, 104)
        self.up_conv3 = double_conv3d(156, 52)
        self.up_conv4 = double_conv3d(78, 26)

        self.final = nn.Conv3d(26, 4, kernel_size=(1, 1, 1))

        self.deep1 = nn.Conv3d(104, 4, kernel_size=(1, 1, 1), padding='same')
        self.deep2 = nn.Conv3d(52, 4, kernel_size=(1, 1, 1), padding='same')

        self.neg_slope = 1e-2
        self.apply(self.InitWeights_He)

    def InitWeights_He(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope, nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def forward(self, input):
        # input shape = b, c, x, y, z
        # ENCODER

        # block1
        x1 = self.down_conv1(input)  #
        x2 = self.max_pool3d(x1)
        # block2
        x3 = self.down_conv2(x2)  #
        x4 = self.max_pool3d(x3)
        x4_d = self.drop(x4)  # dropout
        # block 3
        x5 = self.down_conv3(x4_d)  #
        x6 = self.max_pool3d(x5)
        x6_d = self.drop(x6)  # dropout
        # block 4
        x7 = self.down_conv4(x6_d)  # concat with x
        x8 = self.max_pool3d(x7)
        x8_d = self.drop(x8)  # dropout
        # block 5
        x9 = self.down_conv5(x8_d)

        # DECODER
        # block1
        x = self.upsample3d(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.drop(x)
        x = self.up_conv1(x)
        # block2
        x = self.upsample3d(x)
        x = torch.cat([x, x5], dim=1)
        x = self.drop(x)
        x = self.up_conv2(x)
        ds2 = x
        # block3
        x = self.upsample3d(x)
        x = torch.cat([x, x3], dim=1)
        x = self.drop(x)
        x = self.up_conv3(x)
        ds3_2 = x

        x = self.upsample3d(x)
        x = torch.cat([x, x1], dim=1)
        # x = self.drop(x)
        x = self.up_conv4(x)
        x = self.final(x)

        # Deep supervision
        ds2_1x1_conv = self.deep1(ds2)
        ds1_ds2_sum_upscale = self.upsample3d(ds2_1x1_conv)
        ds3_1x1_conv = self.deep2(ds3_2)
        ds1_ds2_sum_upscale_ds3_sum = torch.add(ds1_ds2_sum_upscale, ds3_1x1_conv)
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample3d(ds1_ds2_sum_upscale_ds3_sum)
        out = torch.add(x, ds1_ds2_sum_upscale_ds3_sum_upscale)

        return out


if __name__ == "__main__":
    model = Unet_3d(0.5).cuda()
    inp = torch.rand(2, 1, 224, 224, 10).cuda()
    output = model(inp)
    # print("Output shape: ", output.shape, "\n")
    # print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.shape)
    print(summary(model, input_size=(1, 224, 224, 10), batch_size=2, device='cuda'))
