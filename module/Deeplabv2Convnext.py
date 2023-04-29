import ever as er

from module.backbone.convnext import convnext_base,convnext_small
import torch.nn as nn
import torch.nn.functional as F
from module.backbone.Encoder import Classifier_Module, PPMBilinear


class Deeplabv2Convnext(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2Convnext, self).__init__(config)
        self.encoder = convnext_base(pretrained=True,in_22k=True)
        if self.config.multi_layer:
            print('Use multi_layer!')
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels // 2, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                        [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24],
                                                      [6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.config.multi_layer:
            if self.config.cascade:
                c3, c4 = self.encoder(x)[-2:]
                x1 = self.layer5(c3)
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                x2 = self.layer6(c4)
                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x2).softmax(dim=1)
            else:
                x = self.encoder(x)[-1]
                x1 = self.layer5(x)
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                x2 = self.layer6(x)
                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x1 + x2).softmax(dim=1)
        else:
            feat, x = self.encoder(x)[-2:]
            # x = self.layer5(x)
            x = self.cls_pred(x)
            # x = self.cls_pred(x)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
            # feat = F.interpolate(feat, (H, W), mode='bilinear', align_corners=True)
            if self.training:
                return x, feat
            else:
                return x.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
                multi_layer = False,
            ),
            inchannels=2048,
            num_classes=7
        ))

if __name__ == '__main__':
    import torch
    model = Deeplabv2Convnext(dict(
        multi_layer=False,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
            fc_dim=1024
        ),
        inchannels=2048,
        num_classes=7
    ))
    x = torch.randn(size=(2,3,512,512))
    model(x)