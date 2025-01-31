import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augs():
    return A.Compose([
        ToTensorV2()
    ])

def get_valid_augs():
    return A.Compose([
        ToTensorV2()
    ])