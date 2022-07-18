import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl


def double_conv2d(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding='same', bias=False),
        nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding='same', bias=False),
        nn.BatchNorm2d(out_c, eps=1e-05, momentum=0.1),
        nn.LeakyReLU(inplace=True)
    )
    return conv


class Unet_2d(pl.LightningModule):
    def __init__(self, drop):
        super(Unet_2d, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True)
        self.drop = nn.Dropout2d(p=drop)
        self.upsample2d = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)

        self.down_conv1 = double_conv2d(1, 48)
        self.down_conv2 = double_conv2d(48, 96)
        self.down_conv3 = double_conv2d(96, 192)
        self.down_conv4 = double_conv2d(192, 384)
        self.down_conv5 = double_conv2d(384, 768)

        self.up_conv1 = double_conv2d(1152, 384)
        self.up_conv2 = double_conv2d(576, 192)
        self.up_conv3 = double_conv2d(288, 96)
        self.up_conv4 = double_conv2d(144, 48)

        self.final = nn.Conv2d(48, 4, kernel_size=(1, 1))

        self.deep1 = nn.Conv2d(192, 4, kernel_size=(1, 1), padding='same')
        self.deep2 = nn.Conv2d(96, 4, kernel_size=(1, 1), padding='same')

        self.neg_slope = 1e-2
        self.apply(self.InitWeights_He)

    def InitWeights_He(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    #   reset the parameters of the model
    # def reset_weights(self, m):
    #     if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or \
    #             isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
    #         m.reset_parameters()

    def forward(self, input_image):
        # input shape = b, c, x, y, z
        # ENCODER
        # block1
        x1 = self.down_conv1(input_image)  #
        x2 = self.max_pool2d(x1)
        # block2
        x3 = self.down_conv2(x2)  #
        x4 = self.max_pool2d(x3)
        x4_d = self.drop(x4)  # dropout
        # block 3
        x5 = self.down_conv3(x4_d)  #
        x6 = self.max_pool2d(x5)
        x6_d = self.drop(x6)  # dropout
        # block 4
        x7 = self.down_conv4(x6_d)  # concat with x
        x8 = self.max_pool2d(x7)
        x8_d = self.drop(x8)  # dropout
        # block 5
        x9 = self.down_conv5(x8_d)

        # DECODER
        # block1
        x = self.upsample2d(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.drop(x)
        x = self.up_conv1(x)
        # block2
        x = self.upsample2d(x)
        x = torch.cat([x, x5], dim=1)
        x = self.drop(x)
        x = self.up_conv2(x)
        ds2 = copy.copy(x)
        # block3
        x = self.upsample2d(x)
        x = torch.cat([x, x3], dim=1)
        x = self.drop(x)
        x = self.up_conv3(x)
        ds3_2 = copy.copy(x)

        x = self.upsample2d(x)
        x = torch.cat([x, x1], dim=1)
        # x = self.drop(x)
        x = self.up_conv4(x)
        x = self.final(x)

        # Deep supervision
        ds2_1x1_conv = self.deep1(ds2)
        ds1_ds2_sum_upscale = self.upsample2d(ds2_1x1_conv)
        ds3_1x1_conv = self.deep2(ds3_2)
        ds1_ds2_sum_upscale_ds3_sum = torch.add(ds1_ds2_sum_upscale, ds3_1x1_conv)
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample2d(ds1_ds2_sum_upscale_ds3_sum)
        out = torch.add(x, ds1_ds2_sum_upscale_ds3_sum_upscale)
        return out


# if __name__ == "__main__":
#     model = Unet_2d(0.1).cuda()
#     inp = torch.rand(8, 1, 240, 240).cuda()
#     output = model(inp)
#     print("output", output.shape, "Number of parameters", sum(p.numel() for p in model.parameters()))
