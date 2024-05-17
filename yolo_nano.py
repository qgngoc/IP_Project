'''
author: lingteng qiu

'''
import os

import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import cv2

from DataHandler import DataLoader
from config import IMAGE_SIZE
from darknet import YOLOLayer,to_cpu

#ACTIVATE_FUNC
from utils.utils import xywh2xyxy, bbox_iou, pad_to_square, resize

ACTIVATE={
    "relu":nn.ReLU,
    "relu6":nn.ReLU6,
    "leaky":nn.LeakyReLU
}
ARCHITECTURE ={
    'layer3':[['EP',150,325,2],['PEP',325,132,325],['PEP',325,124,325],['PEP',325,141,325],
              ['PEP',325,140,325],['PEP',325,137,325],['PEP',325,135,325],['PEP',325,133,325],
              ['PEP',325,140,325]],
    'layer4':[['EP',325,545,2],['PEP',545,276,545],['conv1x1',545,230],['EP',230,489,1],['PEP',489,213,469],
              ['conv1x1',469,189]],
}
YOLO_ARCH = {
    "small": [(116, 90), (156, 198), (373, 326)],
    "middle":[(30, 61), (62, 45), (59, 119)],
    "large":[(10, 13), (16, 30), (33, 23)]
}



class conv3x3(nn.Module):
    def __init__(self,in_planes, out_planes, stride=1, groups=1, dilation=1,act='leaky'):
        super(conv3x3,self).__init__()
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding = 1,dilation=dilation,groups=groups,bias = False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = ACTIVATE[act](inplace = True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class conv1x1(nn.Module):
    def __init__(self,in_planes, out_planes, stride=1, groups=1, dilation=1,act='leaky',use_relu = True):
        super(conv1x1,self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,padding = 0,dilation=dilation,groups=groups,bias = False)
        self.bn = nn.BatchNorm2d(out_planes)
        if use_relu:
            self.relu = ACTIVATE[act](inplace = True)
    def forward(self, x):
        if self.use_relu:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class depth_wise(nn.Module):
    def __init__(self,in_planes,out_planes,stride = 1,act='leaky'):
        super(depth_wise, self).__init__()
        self.conv = nn.Conv2d(in_planes,out_planes,3,stride =stride,padding =1,groups=in_planes,bias =False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = ACTIVATE[act](inplace = True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
class PEP(nn.Module):
    '''
    This is yolo_nano PEP module
    '''
    def __init__(self,in_dim,mid_dim ,out_dim,stride = 1,groups =1,ratios = 2,act='leaky'):
        super(PEP,self).__init__()
        self.conv1X1_0 = conv1x1(in_dim,mid_dim,stride,groups,act=act)
        self.conv1X1_1 = conv1x1(mid_dim,mid_dim*ratios,stride,groups,act=act)
        self.depth_wise = depth_wise(mid_dim*ratios,mid_dim*ratios,1,act=act)
        self.conv1X1_2 = conv1x1(mid_dim*ratios,out_dim,stride,groups,act=act,use_relu=False)
        self.relu = ACTIVATE[act](inplace = True)

        if stride !=1 or in_dim !=out_dim:
            self.downsample = conv1x1(in_dim,out_dim,stride = stride,groups = groups,act=act,use_relu=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1X1_0(x)
        out = self.conv1X1_1(out)
        out = self.depth_wise(out)
        out = self.conv1X1_2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(identity+out)
class FCA(nn.Module):
    '''
    Module structure FCA some like
    '''
    def __init__(self, channels, reduce_channels):
        super(FCA, self).__init__()
        self.channels = channels


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduce_channels, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(reduce_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out


class EP(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1,groups = 1,act = 'leaky'):
        super(EP, self).__init__()
        self.conv1x1_0 = conv1x1(input_channels,output_channels,1,groups,act=act)
        self.depth_wise = depth_wise(output_channels,output_channels,stride,act=act)
        self.conv1x1_1 = conv1x1(output_channels,output_channels,stride = 1,groups=groups,act = act,use_relu= False)
        if stride !=1 or input_channels != output_channels:
            self.downsample = conv1x1(input_channels,output_channels,stride = stride,groups = groups,act=act,use_relu=False)
        else:
            self.downsample = None
        self.relu = ACTIVATE[act](inplace = True)
    def forward(self, x):
        identity = x
        out = self.conv1x1_0(x)
        out = self.depth_wise(out)
        out = self.conv1x1_1(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(identity+out)
class YoloNano(nn.Module):
    '''
    Paper Structure Arch
    return three scale feature,int the paper :(52)4,(26)2,(13)1.
    each channel in here is only 75. for voc 2007 because: (num_class+5)*anchor because voc has 20 classes include background so the
    channel in here is 75
    '''
    def __init__(self,num_class = 20,num_anchor = 3,img_size = 416):
        __FUNC = {
            'EP': EP,
            'PEP': PEP,
            'conv1x1':conv1x1
        }
        super(YoloNano,self).__init__()
        self.num_class =num_class
        self.num_anchor = num_anchor
        self.img_size = img_size
        self.out_channel = (num_class+5)*num_anchor
        self.seen = 0

        self.layer0 = nn.Sequential(conv3x3(3,12,1),conv3x3(12,24,2))
        self.layer1 = nn.Sequential(PEP(24,7,24),
                                    EP(24,70,2),
                                    PEP(70,25,70),
                                    PEP(70,24,70),
                                    EP(70,150,2),
                                    PEP(150,56,150),
                                    conv1x1(150,150,1,1,1,use_relu=True)
                                    )
        self.attention = FCA(150,8)
        self.layer2 = nn.Sequential(PEP(150,73,150),
                                      PEP(150,71,150),
                                      PEP(150,75,150))
        layer3=[]
        for e in ARCHITECTURE['layer3']:
            layer3.append((__FUNC[e[0]](*e[1:])))
        self.layer3 = nn.Sequential(*layer3)
        layer4 = []
        for e in ARCHITECTURE['layer4']:
            layer4.append((__FUNC[e[0]](*e[1:])))
        self.layer4 = nn.Sequential(*layer4)

        #all below this u can change by your self
        self.layer5 = nn.Sequential(PEP(430,113,325),PEP(325,99,207),conv1x1(207,98,use_relu=True))
        #
        self.compress = conv1x1(189,105,use_relu = True)
        self.compress2 = conv1x1(98,47,use_relu=True)



        #Yolo_Layer using to regress the x,y,w,h
        self.scale_4 = nn.Sequential(
            PEP(197,58,122),
            PEP(122,52,87),
            PEP(87,47,93),
            nn.Conv2d(93,self.out_channel,kernel_size=1,stride = 1,padding =0,bias=True)
        )
        self.scale_2 = nn.Sequential(
            EP(98,183,1),
            nn.Conv2d(183,self.out_channel,kernel_size=1,stride = 1,padding =0,bias=True)
        )
        self.scale_1 = nn.Sequential(EP(189,462,1),nn.Conv2d(462,self.out_channel,kernel_size=1,stride = 1,padding =0,bias=True))


        #yolo0 : big_anchor
        #yolo1 : mid_anchor
        #yolo2 : small_anchor
        self.yolo0  = YOLOLayer(YOLO_ARCH['small'], self.num_class,self.img_size )
        self.yolo1  = YOLOLayer(YOLO_ARCH['middle'], self.num_class,self.img_size )
        self.yolo2  = YOLOLayer(YOLO_ARCH['large'], self.num_class,self.img_size )

        self.yolo_layers = [self.yolo0,self.yolo1,self.yolo2]


    def forward(self, x, targets=None, img_scores = None,gt_mix_index = None):


        #img_dim = x.shape[2]
        img_dim = x[0].shape[2]
        loss = 0
        yolo_outputs = []

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.attention(x)
        x_1 = self.layer2(x)
        x_2 = self.layer3(x_1)
        x_3 = self.layer4(x_2)
        x = self.compress(x_3)
        x = F.interpolate(x,scale_factor=2,mode = 'bilinear',align_corners=True)

        x = torch.cat([x,x_2],dim=1)
        x_4 = self.layer5(x)
        x = self.compress2(x_4)
        x = F.interpolate(x,scale_factor=2,mode = 'bilinear',align_corners=True)
        x = torch.cat([x,x_1],dim=1)

        x_scale_4 = self.scale_4(x)
        x_scale_2 = self.scale_2(x_4)
        x_scale_1 = self.scale_1(x_3)


        layer_0_x, layer_loss = self.yolo0(x_scale_1,targets,img_dim,img_scores = img_scores,gt_mix_index=gt_mix_index)
        loss += layer_loss
        yolo_outputs.append(layer_0_x)
        layer_1_x, layer_loss = self.yolo1(x_scale_2,targets,img_dim,img_scores=img_scores,gt_mix_index=gt_mix_index)
        loss += layer_loss
        yolo_outputs.append(layer_1_x)
        layer_2_x, layer_loss = self.yolo2(x_scale_4,targets,img_dim,img_scores=img_scores,gt_mix_index=gt_mix_index)
        loss += layer_loss
        yolo_outputs.append(layer_2_x)

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4,keep_det = 100):
    """
    todo Remeber in here, In yolov3 realize we find the result is kept all.
    todo it is bad in evaluate in early epoch.

    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)


    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        # yolo use MAP maximum a posterior


        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            if len(keep_boxes)>100:
                break
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output


if __name__ == '__main__':
    # x = torch.randn(1,3,416,416)
    imgfolderpath = os.path.dirname(os.path.realpath(__file__)) + "/" + 'wb_localization_dataset' + '/images/train/'
    labelfolderpath = os.path.dirname(
        os.path.realpath(__file__)) + "/" + 'wb_localization_dataset' + '/labels/train/'

    dataloader = DataLoader.DataLoader(imgfolderpath, labelfolderpath)
    train_dataset = dataloader.init_dataset()
    img = train_dataset[0][0] * 255
    image_tensor, a = pad_to_square(img, 255)
    image_tensor = resize(image_tensor, IMAGE_SIZE)
    image_tensor = image_tensor.unsqueeze(0)

    # data = torch.stack([image_tensor])
    # image_tensor = torch.randn(1,3,800,800)
    backbone = YoloNano(num_class=2)
    backbone.train()
    out = backbone(image_tensor)
    # backbone.state_dict()
    # torch.save(backbone.state_dict(),"xixi_a.pth")
    # print('x = ',x)
    print(out.shape)
    output = non_max_suppression(out)[0].int()

    boxes = output[:, :4]
    boxes = np.array(boxes, dtype=int)
    # image = np.array(image_tensor[0].permute(1,2,0), dtype='uint8')
    image = np.array(image_tensor[0].permute(1, 2, 0), dtype='uint8').copy()
    # cv2.imshow("Img", img_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for i in range(len(boxes)):
        box = boxes[i]
        # score = scores[i]
        # label = labels[i]
        # mask = masks[i, 0].cpu().numpy()
        x_start = box[0]
        y_start = box[1]
        x_end = box[2]
        y_end = box[3]
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imshow("Img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print(output)


