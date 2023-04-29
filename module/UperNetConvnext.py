import torch.nn as nn
from module.backbone.convnext import convnext_base
from module.decode.FCNHead import FCNHead
from module.decode.UperHead import UPerHead
import torch.nn.functional as F

class UperNetConvnext(nn.Module):
    def __init__(self):
        super(UperNetConvnext,self).__init__()
        self.backbone = convnext_base(pretrained=True,in_22k=True)
        self.decode_head = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False
        )
        self.auxiliary_head = FCNHead(
            in_channels=512,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
        )

    def forward(self,input):
        h,w = input.shape[2:]
        x = self.backbone(input)
        main_ = self.decode_head(x)
        aux_ = self.auxiliary_head(x) # 输出大小是2，7，32，32
        main_ = F.interpolate(main_,(h,w),mode='bilinear',align_corners=True)
        aux_ = F.interpolate(aux_,(h,w),mode='bilinear',align_corners=True)
        return main_,aux_ # 主分类器，辅助分类器


if __name__ == '__main__':
    import torch
    device = torch.device("cuda")
    model = UperNetConvnext()
    print(model)
    # batchsize % 2==0
    images = torch.rand(size=(2,3,512,512))
    images = images.to(device, dtype=torch.float32)
    model.to(device)
    ret1 = model(images)
    print(model)