import os
import torch
import parser
import models
import data

import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from PIL import Image

if __name__ == '__main__':
    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    save_loader = torch.utils.data.DataLoader(data.DATA(args, mode='save'),
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


    model.eval()
    preds = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs) in enumerate(save_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()

            preds.append(pred)

    preds = np.concatenate(preds)

    ''' add predictions to output_dir'''

    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, pred in enumerate(preds):
        im = Image.fromarray(np.uint8(pred))
        save_path = os.path.join(output_dir, f"{idx:04}.png")
        im.save(save_path)
        print(save_path)
