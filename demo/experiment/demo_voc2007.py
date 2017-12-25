import argparse
import os
import torch
import torch.nn as nn
from copy import deepcopy
from experiment.engine import MultiLabelMAPEngine
from experiment.models import vgg16_sp
from experiment.voc import Voc2007Classification

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--image-size', '-i', default='224', type=str,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
 
def main_voc2007():
    global args, best_prec1, use_gpu 
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval')
    val_dataset = Voc2007Classification(args.data, 'test')
    num_classes = 20

    # load model
    model = vgg16_sp(num_classes, pretrained=True)
    
    print(model)

    criterion = nn.MultiLabelSoftMarginLoss()

    state = {'batch_size': args.batch_size, 'max_epochs': args.epochs, 
            'image_size': args.image_size, 'evaluate': args.evaluate, 'resume': args.resume,
             'lr':args.lr, 'momentum':args.momentum, 'weight_decay':args.weight_decay}
    state['difficult_examples'] = True
    state['save_model_path'] = 'logs/voc2007/'

    engine = MultiLabelMAPEngine(state)
    engine.multi_learning(model, criterion, train_dataset, val_dataset)


if __name__ == '__main__':
    main_voc2007()
