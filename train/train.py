import time

import cv2
import numpy as np
# import torchsummary as summary
import torch
# import torchvision.models.detection.fas
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from tqdm import tqdm

from DataHandler.DataLoader import DataLoader
from config import BASE_PATH, MODEL_PATH


def split_batch(images, targets, batch_size=10):
    splitted = []
    for i in range(0, len(images), batch_size):
        splitted.append((images[i:i + batch_size], targets[i:i + batch_size]))

    return splitted


num_classes = 2
# PATH = 'finetuned_model.pkl'
trainable_backbone_layers = 5
def train():
    # model = fasterrcnn_resnet50_fpn(weights='DEFAULT', trainable_backbone_layers=2)
    model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', trainable_backbone_layers=trainable_backbone_layers)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # model = retinanet_resnet50_fpn(weights='DEFAULT')
    # # in_features = model.head.classification_head.conv[0].in_channels
    # num_anchors = model.head.classification_head.num_anchors
    # model.head.classification_head.num_classes = num_classes
    # model.head = RetinaNetHead(256, num_anchors, num_classes)

    model.train()

    imgfolderpath = BASE_PATH + '/images/train/'
    labelfolderpath = BASE_PATH + '/labels/train/'

    train_loader = DataLoader(imgfolderpath=imgfolderpath, labelfolderpath=labelfolderpath)
    train_dataset = train_loader.init_dataset()
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
    num_episode = 20
    i = 1
    splitted_data = split_batch(full_datax, targets, batch_size=10)
    for episode in range(num_episode):
        for images, targets in splitted_data:
            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(loss for loss in losses.values())
            loss_val = loss.item()
            print(f'Loss at epoch {i}: {loss.item()}')
            i += 1
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
    torch.save(model.state_dict(), MODEL_PATH)

# train()

def eval_one_img(file_path):
    finetuned_model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', trainable_backbone_layers=trainable_backbone_layers)
    in_features = finetuned_model.roi_heads.box_predictor.cls_score.in_features
    finetuned_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # PATH = './input/Model/' + args.DATA_FILE + '/' + ModelSelect + '_finetune_model_embeding.pkl'
    finetuned_model.load_state_dict(torch.load(MODEL_PATH))
    # summary(finetuned_model, (3, 224, 224))
    # print(finetuned_model)
    # finetuned_model = models.vgg16()
    # print(finetuned_model)
    finetuned_model.eval()
    image = cv2.imread(file_path)
    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()
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

# def eval():
#     imgfolderpath = BASE_PATH + '/images/val/'
#     labelfolderpath = BASE_PATH + '/labels/val/'
#
#     test_loader = DataLoader(imgfolderpath=imgfolderpath, labelfolderpath=labelfolderpath)
#     test_dataset = test_loader.init_dataset()
#
#     finetuned_model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT', trainable_backbone_layers=4)
#     in_features = finetuned_model.roi_heads.box_predictor.cls_score.in_features
#     finetuned_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     # PATH = './input/Model/' + args.DATA_FILE + '/' + ModelSelect + '_finetune_model_embeding.pkl'
#     finetuned_model.load_state_dict(torch.load(MODEL_PATH))
#     finetuned_model.eval()


imgfolderpath = BASE_PATH + '/images/val/'
labelfolderpath = BASE_PATH + '/labels/val/'

if __name__ == "__main__":

    loader = DataLoader(imgfolderpath=imgfolderpath, labelfolderpath=labelfolderpath)
    dataset = loader.init_dataset()
    # train()
    # eval()
    eval_one_img('wb_localization_dataset/images/val/nlvnpf-0137-01-045.jpg')