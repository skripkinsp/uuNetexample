import cv2 as cv
import numpy as np
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


def convert_to_grayscale(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return gray_frame


def display_green_objects(frame):
    """
    Выводит исходный кадр и под ним бинарную маску, выделяющую зеленые объекты.

    :param frame: Исходный кадр
    :return: Кадр исходный + бинарная маска зеленых пикселей снизу
    """
    # Преобразование кадра в HSV
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Определение диапазона зеленого цвета в HSV
    lower_red = np.array([80, 150, 50])
    upper_red = np.array([160, 255, 255])

    # Создание маски для зеленых объектов
    green_mask = cv.inRange(hsv_frame, lower_red, upper_red)

    # Применение морфологических операций для удаления артефактов
    kernel = np.ones((15, 15), np.uint8)
    green_mask = cv.morphologyEx(green_mask, cv.MORPH_OPEN, kernel)

    # Объединение исходного кадра и маски
    combined_frame = np.vstack((frame, np.repeat(green_mask[:, :, np.newaxis], 3, axis=2)))
    return combined_frame


for i in range(1000):
    ret, frame = cap.read()
    blue_frame = display_green_objects(frame)
    scale = 0.5
    blue_frame = cv.resize(blue_frame, (-1, -1), fy=scale, fx=scale)
    cv.imshow('cam', blue_frame)
    if cv.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()