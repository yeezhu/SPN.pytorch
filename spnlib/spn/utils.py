import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.misc import imresize
from scipy.ndimage import label
from .modules import SoftProposal

# helper functions

def hook_spn(model):
    if not (hasattr(model, 'sp_hook') and hasattr(model, 'fc_hook')):
        model._training = model.training
        model.train(False)
        
        def _sp_hook(self, input, output):
            self.parent_modules[0].class_response_maps = output
        def _fc_hook(self, input, output):
            if hasattr(self.parent_modules[0], 'class_response_maps'):
                self.parent_modules[0].class_response_maps = F.conv2d(self.parent_modules[0].class_response_maps, self.weight.unsqueeze(-1).unsqueeze(-1))
            else:
                raise RuntimeError('The SPN is broken, please recreate it.')
                
        sp_layer = None
        fc_layer = None
        for mod in model.modules():
            if isinstance(mod, SoftProposal):
                sp_layer = mod
            elif isinstance(mod, torch.nn.Linear):
                fc_layer = mod
        
        if sp_layer is None or fc_layer is None:
            raise RuntimeError('Invalid SPN model')
        else:
            sp_layer.parent_modules = [model]
            fc_layer.parent_modules = [model]
            model.sp_hook = sp_layer.register_forward_hook(_sp_hook)
            model.fc_hook = fc_layer.register_forward_hook(_fc_hook)
    
    return model

def unhook_spn(model):
    try:
        model.sp_hook.remove()
        model.fc_hook.remove()
        del model.sp_hook
        del model.fc_hook
        model.train(model._training)
        return model
    except:
        raise RuntimeError('The model haven\'t been hooked!')

def compute_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(x_b - x_a + 1, 0) * max(y_b - y_a + 1, 0)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    return inter_area / float(box_a_area + box_b_area - inter_area)

def bbox_nms(bbox_list, threshold=0.5):
    bbox_list = sorted(bbox_list,  key=lambda x: x[-1], reverse=True)
    selected_bboxes = []
    while len(bbox_list) > 0:
        obj = bbox_list.pop(0)
        selected_bboxes.append(obj)
        def iou_filter(x):
            iou = compute_iou(obj[1:5], x[1:5])
            if (x[0] == obj[0] and iou >= threshold):
                return None
            else:
                return x
        bbox_list = list(filter(iou_filter, bbox_list))
    return selected_bboxes

def gen_filter(bbox_threshold=(0., 50)):
    def _filter(x):
        xmin, ymin, xmax, ymax = x[1:5]
        w, h = (xmax - xmin), (ymax - ymin)
        if x[-1] > bbox_threshold[0] and w >= bbox_threshold[1] and h >= bbox_threshold[1]:
            return x
        else:
            return None
    return _filter

def extract_bbox_from_map(input):
    assert input.ndim == 2, 'Invalid input shape'
    rows = np.any(input, axis=1)
    cols = np.any(input, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def extract_point_from_map(input):
    assert input.ndim == 2, 'Invalid input shape'
    cols = input.shape[1]
    index = np.argmax(input)
    return index % cols, index // cols

def localize_from_map(class_response_map, class_idx=0, location_type='bbox', threshold_ratio=1, multi_objects=True):
    assert location_type == 'bbox' or location_type == 'point', 'Unknown location type'
    foreground_map = class_response_map >= (class_response_map.mean() * threshold_ratio)
    if multi_objects:
        objects, count = label(foreground_map)
        res = []
        for obj_idx in range(count):
            obj = objects == (obj_idx + 1)
            if location_type == 'bbox':
                score = class_response_map[obj].mean()
                extraction = extract_bbox_from_map
            elif location_type == 'point':
                obj = class_response_map * obj.astype(float)
                score = np.max(obj)
                extraction = extract_point_from_map
            res.append((class_idx,) + extraction(obj) + (score,))
        return res
    else:
        if location_type == 'bbox':
            return [(class_idx,) + extract_bbox_from_map(foreground_map) + (class_response_map.mean(),), ]
        elif location_type == 'point':
            return [(class_idx,) + extract_point_from_map(class_response_map) + (class_response_map.max(),), ]

def object_localization(models, input, **kwargs):
    # multi-scale detection
    scales = sorted(kwargs.pop('scales', models.keys()), reverse=True)
    # localize with/without prediction
    pred_labels = kwargs.pop('gt_labels', None)
    # classification threshold
    threshold = kwargs.pop('threshold', 0)
    # switch to inference mode
    force_inference = kwargs.pop('force_inference', True)
    # NMS threshold
    nms_threshold = kwargs.pop('nms_threshold', 0.)
    # type of localization
    location_type = kwargs.get('location_type', 'bbox')

    assert len(models) == 1 or set(scales).issubset(set(models.keys())), 'Invalid scales'
    
    if input.ndimension() == 3: 
        input = input.unsqueeze(0)
    assert input.size(0) == 1, 'Batch processing is currently not supported'

    # enable spn inference mode
    if force_inference: 
        models = {k:hook_spn(v) for k, v in models.items()}

    # localize objects
    predictions = []
    for size in scales:
        model = models[size] if len(models) > 1 else next(iter(models.values()))
        class_scores = model(F.upsample(input, size=(size, size), mode='bilinear'))
        pred_labels = torch.nonzero(class_scores.data.squeeze() > threshold).squeeze() if pred_labels is None else pred_labels
        for class_idx in pred_labels:
            kwargs['class_idx'] = class_idx
            class_response_map = F.upsample(model.class_response_maps[0, class_idx].unsqueeze(0).unsqueeze(0), size=(input.size(2), input.size(3)), mode='bilinear')
            predictions += localize_from_map(class_response_map.squeeze().data.cpu().numpy(), **kwargs)
    
    # non maximum suppression
    if location_type == 'bbox' and len(models) > 1:
        predictions = list(filter(gen_filter(), bbox_nms(predictions, nms_threshold))) 

    return predictions, pred_labels