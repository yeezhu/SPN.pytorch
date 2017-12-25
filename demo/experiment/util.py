import os
import re
import math
import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from urllib.request import urlretrieve

import experiment.models as models
from evaluation.SP_GoogLeNet import SP_GoogLeNet

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


def download_url(url, destination=None, progress_bar=True):
    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

# load ground truth
def load_ground_truth_voc(data_root, split):
    ground_truth = dict(
        image_list = [],
        image_sizes = [],
        gt_labels = [],
        gt_bboxes = [],
        class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    )
    # load image list
    with open(os.path.join(data_root, 'ImageSets', 'Main', split + '.txt')) as f:
        img_list = f.readlines()
    ground_truth['image_list'] = [line.strip() for line in img_list]
    
    for c in ground_truth['class_names']:
        cls_labels = [int(l.split(' ')[-1]) for l in list(open(os.path.join(data_root, 'ImageSets', 'Main', c + '_' + split + '.txt')))]
        ground_truth['gt_labels'].append(cls_labels)
    ground_truth['gt_labels'] = np.array(ground_truth['gt_labels']).T
        
    # load annotations
    for idx, img_name in enumerate(ground_truth['image_list']):
        with open(os.path.join(data_root, 'Annotations', img_name + '.xml')) as f:
            anno = BeautifulSoup(''.join(f.readlines()), "xml")
        ground_truth['image_sizes'].append((int(anno.find('size').height.contents[0]), int(anno.find('size').width.contents[0])))
        bboxes = anno.findAll('object')
        for bbox_idx, bbox in enumerate(bboxes):
            category = ground_truth['class_names'].index(str(bbox.find('name').contents[0]))
            ground_truth['gt_bboxes'].append((idx, category, int(bbox.xmin.contents[0]), int(bbox.ymin.contents[0]), int(bbox.xmax.contents[0]), int(bbox.ymax.contents[0])))
    
    ground_truth['gt_bboxes'] = np.array(ground_truth['gt_bboxes'])                          
    return ground_truth

def load_ground_truth_imagenet(data_root):
    ground_truth = dict(
        image_list = [],
        black_list = [],
        image_sizes = [],
        gt_labels = [],
        gt_bboxes = [],
        class_names = [],
        class_words = [],
    )
    # load image list
    ground_truth['image_list'] = [line.split('.')[0] for line in list(open(os.path.join(data_root, 'imagenet_val.txt'),'r'))]
    # load black list
    ground_truth['black_list'] = [int(x.strip()) for x in list(open(os.path.join(data_root, 'ILSVRC2014_clsloc_validation_blacklist.txt'), 'r'))]

    for item in list(open(os.path.join(data_root, 'synset_words.txt'),'r')):
        a = re.match(r'^(\w\d+) (.+)', item)
        ground_truth['class_words'].append(a.group(1))
        ground_truth['class_names'].append(a.group(2))
    
    # load annotations
    for idx, img_name in enumerate(ground_truth['image_list']):
        with open(os.path.join(data_root, 'val', img_name + '.xml')) as f:
            anno = BeautifulSoup(''.join(f.readlines()), "xml")
        ground_truth['image_sizes'].append((int(anno.find('size').height.contents[0]), int(anno.find('size').width.contents[0])))
        bboxes = anno.findAll('object')
        for bbox_idx, bbox in enumerate(bboxes):
            category = ground_truth['class_words'].index(str(bbox.find('name').contents[0]))
            ground_truth['gt_bboxes'].append((idx, category, int(bbox.xmin.contents[0]), int(bbox.ymin.contents[0]), int(bbox.xmax.contents[0]), int(bbox.ymax.contents[0])))
        
        ground_truth['gt_labels'].append(category)
    ground_truth['gt_bboxes'] = np.array(ground_truth['gt_bboxes'])                          
    return ground_truth

def load_image_voc(image_name):
    image_raw = Image.open(image_name).convert("RGB") 
    image_normalized = torch.from_numpy(np.array(image_raw)).permute(2, 0, 1).cuda().float()
    image_normalized = image_normalized.index_select(0, torch.LongTensor([2,1,0]).cuda())   
    image_normalized = (image_normalized - torch.Tensor([103.939, 116.779, 123.68]).cuda().view(3, 1, 1))
    input_var = Variable(image_normalized.unsqueeze(0), volatile=True)
    return image_raw, input_var

def load_image_imagenet(image_name, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image_raw = Image.open(image_name).convert("RGB") 
    image_normalized = torch.from_numpy(np.array(image_raw)).permute(2, 0, 1).cuda().float() / 255.0
    image_normalized = (image_normalized - torch.Tensor(mean).cuda().view(3, 1, 1)) \
        / torch.Tensor(std).cuda().view(3, 1, 1)
    input_var = Variable(image_normalized.unsqueeze(0), volatile=True)
    return image_raw, input_var

def load_model_voc(model_path, multiscale=False, scale=224):
    model = models.vgg16_sp(20, False)
    state_dict = torch.load(model_path)
    model_dict = {}
    for sdict in state_dict:
        model.load_state_dict(sdict['state_dict'])
        model_dict[sdict['image_size']] = deepcopy(model).cuda()
    return model_dict if multiscale else {scale: model_dict[scale]}

def load_model_imagenet(filename, scale=224):
    model = SP_GoogLeNet(state_dict=filename).cuda()
    model.inference()
    return {scale: model}

def pointing(pred_points, ground_truth, with_prediction=False, scores=None, tol=15):
    class_acc = np.zeros(shape=(len(ground_truth['class_names']),1))
    gt_bboxes = ground_truth['gt_bboxes']
    for c, cls in enumerate(ground_truth['class_names']):
        cls_pred_points = pred_points[pred_points[:,1] == c, :]
        cls_gt_bboxes = gt_bboxes[gt_bboxes[:,1] == c, :]    
        gt_labels = ground_truth['gt_labels'][:,c]
        cls_inds = (gt_labels >= 0).nonzero()
        hit = np.zeros(shape=(len(gt_labels),)) 
        for cidx in cls_inds[0]:
            pred = cls_pred_points[cls_pred_points[:,0]==cidx,2:4].squeeze()
            if len(pred) > 0:
                gt = cls_gt_bboxes[cls_gt_bboxes[:,0]==cidx,2:]
                gt_aug = gt + np.array([-tol, -tol, tol, tol])            
                containMax = (pred[0] >= gt_aug[:,0]) & (pred[0] <= gt_aug[:,2]) & \
                    (pred[1] >= gt_aug[:,1]) & (pred[1] <= gt_aug[:,3])
                if sum(containMax) > 0:
                    hit[cidx] = 1
                else:
                    hit[cidx] = -1
        class_acc[c] = sum(hit == 1) / len(cls_inds[0])
    return class_acc.mean()

def ious(pred, gt):
    numObj = len(gt)
    gt = np.tile(gt,[len(pred),1])
    pred = np.repeat(pred, numObj, axis=0)
    bi = np.minimum(pred[:,2:],gt[:,2:]) - np.maximum(pred[:,:2], gt[:,:2]) + 1
    area_bi = np.prod(bi.clip(0), axis=1)
    area_bu = (gt[:,2] - gt[:,0] + 1) * (gt[:,3] - gt[:,1] + 1) + (pred[:,2] - pred[:,0] + 1) * (pred[:,3] - pred[:,1] + 1) - area_bi
    return area_bi / area_bu

def corloc(pred_boxes, ground_truth):
    class_corloc = []
    gt_bboxes = ground_truth['gt_bboxes']
    for c, cls in enumerate(ground_truth['class_names']):
        cls_pred_boxes = pred_boxes[pred_boxes[:,1] == c, :]
        cls_gt_bboxes = gt_bboxes[gt_bboxes[:,1] == c, :]
        cls_inds = (ground_truth['gt_labels'][:,c] == 1).nonzero()
        cor = 0
        for cidx in cls_inds[0]:
            pred = cls_pred_boxes[cls_pred_boxes[:,0]==cidx,2:6]
            if len(pred) > 0:
                gt = cls_gt_bboxes[cls_gt_bboxes[:,0]==cidx,2:]                
                if max(ious(pred, gt)) >= 0.5:
                    cor += 1
        class_corloc.append(cor/len(cls_inds[0]))   
    return sum(class_corloc)/len(class_corloc)

def locerr(pred_boxes, ground_truth):
    gt_bboxes = ground_truth['gt_bboxes']
    err = 0
    for idx in range(len(ground_truth['image_list'])):
        if idx+1 not in ground_truth['black_list']:
            pred = pred_boxes[pred_boxes[:,0]==idx, 2:6]
            gt = gt_bboxes[gt_bboxes[:,0]==idx, 2:]
            if max(ious(pred, gt)) < 0.5:
                err += 1
    return err / (len(ground_truth['image_list']) - len(ground_truth['black_list']))

def draw_points(img, points, class_names, length=5, width=3,  font_size=20, color=(255, 255, 0)):
    img = img.copy()
    fnt = ImageFont.truetype('arial.ttf', font_size)
    for point in points:
        class_idx, x, y, score = point
        draw = ImageDraw.Draw(img)
        draw.line((x - length, y - length, x + length , y + length), fill=color, width=width)
        draw.line((x + length, y - length, x - length , y + length), fill=color, width=width)
        draw.text((x - length, y + length), '{}({:.2f})'.format(class_names[int(class_idx)], score), font=fnt, fill=color)
    return img

def draw_bboxes(img, bboxes, class_names, width=3, font_size=20, color=(255, 255, 0)):
    img = img.copy()
    fnt = ImageFont.truetype('arial.ttf', font_size)
    for bbox in bboxes:
        class_idx, xmin, ymin, xmax, ymax, score = bbox
        draw = ImageDraw.Draw(img)
        draw.line((xmin, ymin, xmax, ymin), fill=color, width=width)
        draw.line((xmax, ymin, xmax, ymax), fill=color, width=width)
        draw.line((xmin, ymax, xmax, ymax), fill=color, width=width)
        draw.line((xmin, ymin, xmin, ymax), fill=color, width=width)
        draw.text((xmin, ymin), '{}({:.2f})'.format(class_names[int(class_idx)], score), font=fnt, fill=color)
    return img