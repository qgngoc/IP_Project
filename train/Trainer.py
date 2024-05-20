import time

import sys
import os

parent_dir = os.path.dirname(os.path.realpath(__file__)) + "/../"
sys.path.append(parent_dir)

import cv2
import numpy as np
# import torchsummary as summary
import torch
# import torchvision.models.detection.fas
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from tqdm import tqdm
from DataHandler.DataLoader import DataLoader
from DataHandler.DataPreprocessor import DataPreprocessor
from config import BASE_PATH, MODEL_PATH, split_batch

from ultralytics import YOLO
# model = YOLO('../best.pt')
num_classes = 2


class Trainer:
    def __init__(self, model_path='../best.pt'):
        self.model = YOLO(model_path)

    @staticmethod
    def create_model(model):
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # TODO customizing model backbone
        return model

    def train(self):
        model = self.model
        # hyp = dict()
        # hyp['flipud'] = 1
        # hyp['fliplr'] = 1
        model.train(data="hannom.yaml", epochs=30,
                    imgsz=900,
                    batch=8,
                    optimizer='RMSProp',   # Adam, AdamW
                    lr0=0.001,
                    momentum=0.9
                    # fliplr=1,
                    # flipud=1
                    )

        path = model.export(format="ONNX")
        print(path)

    def eval_one_img(self, procesed_image):
        finetuned_model = self.model  # torch.load(MODEL_PATH)
        time1 = time.time()
        result = finetuned_model([procesed_image])[0]
        print(time.time() - time1)
        for confidence, box in zip(result.boxes.conf, result.boxes.xyxy):
            x_start, y_start, x_end, y_end = box
            x_start = int(x_start)
            y_start = int(y_start)
            x_end = int(x_end)
            y_end = int(y_end)
            cv2.rectangle(procesed_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.imshow("Img", procesed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def evaluate(self, img, threshold=0.2):
        finetuned_model = self.model
        result = finetuned_model([img], verbose=False)[0]
        preds = []
        for confidence, box in zip(result.boxes.conf, result.boxes.xywh):
            # confidence = confidence.item()
            if confidence <= threshold:
                continue
            confidence = 1.0
            x_center, y_center, width, height = box
            x_center = int(x_center)
            y_center = int(y_center)
            width = int(width)
            height = int(height)
            preds += [(confidence, x_center, y_center, width, height)]

        return np.array(preds)


if __name__ == "__main__":
    trainer = Trainer('yolov8n.pt')
    trainer.train()
    BASE_PATH = "/Users/quangngoc0811/Documents/UETFiles/IP/IP_Project/wb_localization_dataset"
    validate_dataset = DataLoader(imgfolderpath=BASE_PATH + '/images/val/',
                                  labelfolderpath=BASE_PATH + '/labels/val/').init_dataset()
    test_data = validate_dataset[0]
    test_img = test_data[0]

    trainer.eval_one_img(test_img)

