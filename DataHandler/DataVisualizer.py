import os

import cv2
import numpy as np
import torch

class DataVisualizer:
    def __init__(self):
        pass

    def visualize_data(self, data):
        image, label = data
        image = np.array(image, dtype='uint8')
        cv2.imshow("Img", image)
        # cv2.waitKey(0)
        for box in label:
            _, x_center, y_center, width, height = box
            width *= image.shape[1]
            height *= image.shape[0]
            x_start = int(x_center*image.shape[1] - width/2)
            y_start = int(y_center*image.shape[0] - height/2)
            x_end = int(x_start + width)
            y_end = int(y_start + height)
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)


        cv2.imshow("Img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()