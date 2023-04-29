# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import ConvModule
import torch.nn as nn
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False,
            ))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False,
                ))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                inplace=False,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs) # 拿到H/16，W/16，512 这个特征。进行 convs 操作，具体的操作步骤如下：
        output = self.convs(x) # Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 
                               # + BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                               # + ReLU()
                               # 得到 output

        if self.concat_input: # concat_input=False 未使用 self.conv_cat 这块代码
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output) # 卷积 Conv2d(256, 7, kernel_size=(1, 1), stride=(1, 1)) 得到 output
        return output

if __name__ == '__main__':
    model = FCNHead(in_channels=128,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False)
    print(model)
    images1 = torch.randn(size=(2,128,128,128))
    images2 = torch.randn(size=(2,128,64,64))
    images3 = torch.randn(size=(2,128,32,32))
    images4 = torch.randn(size=(2,128,16,16))
    img = [images1,images2,images3,images4]
    retval = model(img)
    lable = torch.randn(size=(2,7,32,32))
    import torch.nn.functional as F
    d = F.cross_entropy(retval,lable)
    d.backward()
    print(d)