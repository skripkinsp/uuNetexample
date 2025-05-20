import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class SAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Загрузка маски (0-255)

        # Нормализация
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float() / 255.0  # [0, 1]

        return image, mask.unsqueeze(0)  # Добавляем размерность канала

# Создание DataLoader
dataset = SAMDataset("dataset/images", "dataset/masks")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
