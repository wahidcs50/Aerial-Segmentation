import os
import cv2
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def load_and_patch_data(root_directory, patch_size):
    scaler = MinMaxScaler()
    image_dataset = []
    mask_dataset = []

    for path, subdirs, files in os.walk(root_directory):
        dirname = path.split(os.path.sep)[-1]
        if dirname == 'images':
            images = os.listdir(path)
            images.sort()
            for image_name in images:
                if image_name.endswith(".jpg"):
                    image = cv2.imread(os.path.join(path, image_name), 1)
                    SIZE_X = (image.shape[1] // patch_size) * patch_size
                    SIZE_Y = (image.shape[0] // patch_size) * patch_size
                    image = Image.fromarray(image)
                    image = image.crop((0, 0, SIZE_X, SIZE_Y))
                    image = np.array(image)
                    patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            single_patch_img = patches_img[i, j, 0, :, :]
                            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                            image_dataset.append(single_patch_img)

        elif dirname == 'masks':
            masks = os.listdir(path)
            masks.sort()
            for mask_name in masks:
                if mask_name.endswith(".png"):
                    mask = cv2.imread(os.path.join(path, mask_name), 1)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    SIZE_X = (mask.shape[1] // patch_size) * patch_size
                    SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                    mask = Image.fromarray(mask)
                    mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                    mask = np.array(mask)
                    patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
                    for i in range(patches_mask.shape[0]):
                        for j in range(patches_mask.shape[1]):
                            single_patch_mask = patches_mask[i, j, 0, :, :]
                            mask_dataset.append(single_patch_mask)

    return np.array(image_dataset), np.array(mask_dataset)