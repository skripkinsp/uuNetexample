import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Загрузка обученной модели (или предобученной 'yolov8n-seg.pt')
model = YOLO('runs/segment/train/weights/best.pt')

# Загрузка изображения
image_path = "test.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

# Предсказание
results = model.predict(image_path, conf=0.5)  # conf - порог уверенности

# Получение масок
masks = results[0].masks  # Все маски на изображении

if masks is not None:
    # Создаем пустую маску (черный фон)
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i, mask in enumerate(masks.data):
        # Конвертируем тензор маски в numpy array
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # 0 или 255
        resized_mask = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
        
        # Наложение маски (можно использовать разные цвета)
        color_mask = np.zeros_like(image)
        color_mask[:, :, 0] = resized_mask  # Синий канал (BGR)
        
        # Добавляем маску на общее изображение
        combined_mask = cv2.bitwise_or(combined_mask, resized_mask)
        
        # Визуализация каждой маски
        plt.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
        plt.title(f"Маска объекта {i + 1}")
        plt.axis('off')
        plt.show()

    # Сохранение общей маски (бинарная)
    cv2.imwrite("combined_mask.png", combined_mask)

    # Наложение маски на исходное изображение
    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
