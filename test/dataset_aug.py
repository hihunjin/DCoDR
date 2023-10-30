from datasets.builder import build_dataset
from transforms_utils import AUGS_DICT
from dataset import Images_Augmentation_Subset


dataset = build_dataset(
    _target_="two_class_color_mnist",
    split="train",
    transform=None,
    target_type="attr",
)

pos_transforms = [
    AUGS_DICT['gblurr'],
    AUGS_DICT['high_contrast'],
    AUGS_DICT['crop'],
    AUGS_DICT['high_satur']
]

temp_dataset = Images_Augmentation_Subset(
    orig_dataset=dataset,
    pos_augments=pos_transforms,
    num_pos=1,
)
temp_dataset[0]
len
