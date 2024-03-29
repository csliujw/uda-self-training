# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.psp_head import PPM

from mmseg.ops import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class UPerHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.


    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(
            input_transform='multiple_select',  **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            inplace=False)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            inplace=False)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        # psp_outs 可以说是中间特征，应该做中间特征的对齐。
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""
        # print(inputs.shape)
        inputs = self._transform_inputs(inputs) # inputs 四个层级的特征
        # input 做个上采样，返回咋样

        # 1*1 卷积将 backbone 提取到的特征从128-->512   256-->512  512-->512， 最后一层 用 PPM。
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 最后一层（H/32，W/32，1024）做PPM
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]

            laterals[i - 1] = laterals[i - 1]+resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs) # 2，512，128，128
        output = self.cls_seg(output)
        return output

if __name__ == '__main__':
    model = UPerHead(in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False)
    print(model)
    images1 = torch.randn(size=(2,128,128,128))
    images2 = torch.randn(size=(2,256,64,64))
    images3 = torch.randn(size=(2,512,32,32))
    images4 = torch.randn(size=(2,1024,16,16))
    img = [images1,images2,images3,images4]
    retval = model(img)
    # lable = torch.randn(size=(2,7,128,128))
    # import torch.nn.functional as F
    # d = F.cross_entropy(retval,lable)
    # d.backward()
    # print(d)