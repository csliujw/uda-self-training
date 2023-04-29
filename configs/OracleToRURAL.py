from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale, Resize
from albumentations import OneOf, Compose,RandomScale,Resize
import ever as er


TARGET_SET = 'RURAL'

BASE = "./"

train_dir = dict(
    image_dir=[
        BASE+'LoveDA/Train/Rural/images_png/',
    ],
    mask_dir=[
        BASE+'LoveDA/Train/Rural/masks_png/',
    ],
)
val_dir = dict(
    image_dir=[
        BASE+'LoveDA/Val/Rural/images_png/',
    ],
    mask_dir=[
        BASE+'LoveDA/Val/Rural/masks_png/',
    ],
)

test_dir = dict(
    image_dir=[
        BASE+'LoveDA/Test/Rural/images_png/',
    ]
)

size=512

SOURCE_DATA_CONFIG = dict(
    image_dir=train_dir['image_dir'],
    mask_dir=train_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(size,size),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=16,
)


EVAL_DATA_CONFIG = dict(
    image_dir=val_dir['image_dir'], 
    mask_dir=val_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=8,
    num_workers=8,
)

TEST_DATA_CONFIG = dict(
    image_dir=test_dir['image_dir'],
    mask_dir=test_dir['image_dir'],
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()
    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=8,
    num_workers=8,
)
