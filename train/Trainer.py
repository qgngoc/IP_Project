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
from yolo_nano import YoloNano

imgfolderpath = BASE_PATH + '/images/train/'
labelfolderpath = BASE_PATH + '/labels/train/'

#DEFAULT_MODEL = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', trainable_backbone_layers=5)
# DEFAULT_MODEL.load_state_dict(torch.load('/home/phong/VScode/IP_Project/yolo_nano_22.4_40.7.pth'))
model = YoloNano(num_class=2)#torch.load('yolo_nano_0.557.pth', map_location=torch.device('cpu'))
model.load_state_dict(torch.load('yolo_nano_0.557.pth', map_location=torch.device('cpu')), strict=False)
#print(model)
num_classes = 2


class Trainer:
    def __init__(self, pretrained_model=model):
        self.model = self.create_model(pretrained_model)
        self.dataset = DataLoader(imgfolderpath=imgfolderpath, labelfolderpath=labelfolderpath).init_dataset()

    @staticmethod
    def create_model(model):
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # TODO customizing model backbone
        
        return model

    def train(self):
        # model = fasterrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=2)
        model = self.model
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # model = retinanet_resnet50_fpn(weights='DEFAULT')
        # # in_features = model.head.classification_head.conv[0].in_channels
        # num_anchors = model.head.classification_head.num_anchors
        # model.head.classification_head.num_classes = num_classes
        # model.head = RetinaNetHead(256, num_anchors, num_classes)

        model.train()


        train_dataset = self.dataset
        targets = []
        full_datax = []
        labels = torch.randint(1, 91, (4, 11))
        b = torch.rand(4, 11, 4)
        a = labels[0]
        for data in train_dataset:
            datax, datay = data
            _, boxes = datay
            labels = [1] * list(boxes.size())[0]
            d = {
                'boxes': boxes,
                'labels': torch.tensor(labels, dtype=torch.int64)
            }
            targets.append(d)
            full_datax.append(datax)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.ASGD(params, lr=0.005,
                                    # momentum=0.9,
                                     weight_decay=0.0005)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                step_size=3,
        #                                                gamma=0.1)
        num_epoches = 6
        i = 1
        splitted_data = split_batch(full_datax, targets, batch_size=5)
        for epoch in range(num_epoches):
            for images, targets in splitted_data:
                optimizer.zero_grad()
                losses = model(images, targets)
                loss = sum(loss for loss in losses.values())
                loss_val = loss.item()
                print(f'Loss at step {i}: {loss.item()}')
                i += 1
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
        #torch.save(model.state_dict(), MODEL_PATH)
        torch.save(model, MODEL_PATH)
    # train()

    def eval_one_img(self, procesed_image):
        finetuned_model = self.model #torch.load(MODEL_PATH)

        finetuned_model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')), strict = False)

        finetuned_model.eval()
        image = procesed_image * 255
        image = np.array(image.permute(1,2,0), dtype='uint8')
        # edge_filtered_img = DataPreprocessor.edge_filtering(image)
        image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
        
        print(image_tensor.shape)
        
        time1 = time.time()
        with torch.no_grad():
            predictions = finetuned_model([image_tensor])

        print(f'Run time: {time.time()-time1} second(s)')

        result = predictions[0]
        # masks = result['masks']
        scores = result['scores'].tolist()
        boxes = result['boxes']
        labels = result['labels'].tolist()
        boxes = np.asarray(boxes, dtype=int)
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            label = labels[i]
            # mask = masks[i, 0].cpu().numpy()
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def evaluate(self, img, threshold=0.5):
        # img = cv2.imread(file_path)
        img_tensor = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        self.model.eval()
        preds = []
        with torch.no_grad():
            predictions = self.model([img_tensor])
            result = predictions[0]
            # masks = result['masks']
            scores = result['scores'].tolist()
            boxes = result['boxes']
            # labels = result['labels'].tolist()
            boxes = np.asarray(boxes, dtype=int)
            for i in range(len(boxes)):
                box = boxes[i]
                confidence = scores[i]
                if confidence < threshold:
                    continue
                x_start = box[0]
                y_start = box[1]
                x_end = box[2]
                y_end = box[3] 
                width = x_end - x_start
                height = y_end - y_start
                x_center = int(x_start + width / 2)
                y_center = int(y_start + height / 2)
                preds += [(confidence, x_center, y_center, width, height)]

        return np.array(preds)


if __name__ == "__main__":

    trainer = Trainer()
    #trainer.train() 

    validate_dataset = DataLoader(imgfolderpath=BASE_PATH + '/images/train/', labelfolderpath=BASE_PATH + '/labels/train/').init_dataset()
    test_data = validate_dataset[4]
    test_img = test_data[0]

    trainer.eval_one_img(test_img)

