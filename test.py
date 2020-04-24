import os
import torch

import parser
import models
import data
from torch.utils.data import Dataset

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mean_iou_evaluate import mean_iou_score




def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)



    return mean_iou_score(gts, preds)

if __name__ == '__main__':

    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=False)
    ''' prepare mode '''
    if args.model=='Net' :
        model = models.Net(args)
        model.cuda() # load model to gpu
    if args.model=='Net_improved' :
        model = models.Net_improved(args)
        model.cuda() # load model to gpu

    ''' resume save model '''

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader)
    print('Testing Accuracy: {}'.format(acc))
