import os
import nibabel as nib
import numpy as np
from PIL import Image

def convert_2d_to_3d_nifti(input_img_path, output_nifti_path):
    img = np.array(Image.open(input_img_path))
    if len(img.shape) == 2:  # Grayscale
        img = img[..., np.newaxis]  # (H, W, 1)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
        img = img[..., np.newaxis]  # (H, W, 3, 1) - nnUNet обработает это правильно
    nii_img = nib.Nifti1Image(img, affine=np.eye(4))
    nib.save(nii_img, output_nifti_path)

# Пример использования:
base_dir = "dataset"
output_dir = "nnUNet_raw/Dataset001_MyTask"

os.makedirs(f"{output_dir}/imagesTr", exist_ok=True)
os.makedirs(f"{output_dir}/labelsTr", exist_ok=True)

# Конвертируем изображения
for i, img_name in enumerate(os.listdir(f"{base_dir}/train_frames")):
    convert_2d_to_3d_nifti(
        f"{base_dir}/train_frames/{img_name}",
        f"{output_dir}/imagesTr/case_{i:04d}.nii.gz"
    )

# Конвертируем маски (должны быть одноканальными)
for i, mask_name in enumerate(os.listdir(f"{base_dir}/train_frames")):
    mask = np.array(Image.open(f"{base_dir}/train_masks/{mask_name}"))
    mask = (mask > 0).astype(np.uint8)  # Бинаризация, если маска не 0/1
    nii_mask = nib.Nifti1Image(mask[..., np.newaxis], affine=np.eye(4))
    nib.save(nii_mask, f"{output_dir}/labelsTr/case_{i:04d}.nii.gz")