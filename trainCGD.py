import json
import os
import sys
import time
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device_ids = [0, 1, 2, 3]

import torch.nn.functional as F
from torch import optim, nn
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from torch.optim import lr_scheduler
from tqdm import tqdm
from Datasets.lvDataset import lvDataset
from Models.CGDnet import CBGModel
from loss import BatchHardTripletLoss, LabelSmoothingCrossEntropyLoss
import torch
import numpy as np

criterion = nn.CrossEntropyLoss()


def get_train_loader(DataSet, batchSize):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=180),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        normalize])


    train_set = DataSet(train=True, transform_train=transform_train)
    train_loader = torchdata.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=2,
                                        pin_memory=False, drop_last=True)

    return train_loader



def get_val_loader(DataSet, batchSize):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_val= transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])

    val_set = DataSet(train=False, transform_val = transform_val)
    val_loader = torchdata.DataLoader(val_set, batch_size=batchSize, shuffle=True, num_workers=2,
                                        pin_memory=False, drop_last=True)

    return val_loader



def train():


    print('===start training===')
    model.train()

    print("Train Epoch: {}".format(epoch))
    total_loss = 0

    progressbar = tqdm(range(len(train_loader)))
    for batch_id, (img, class_label) in enumerate(train_loader):
        img, class_label = [x.cuda() for x in (img, class_label)]
        # net.zero_grad()
        optimizer.zero_grad()
        if img.shape[0] < params['batchSize']:
            continue
        if batch_id == 0:
            print("batch_id")

        features, classes = model(img)
        class_loss = class_criterion(classes, class_label)
        feature_loss = feature_criterion(features, class_label)
        loss = class_loss + feature_loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        progressbar.set_description(" loss:{:.4f}".format(loss.item()))
        progressbar.update(1)
    progressbar.close()

    scheduler.step()  # 每个epoch里用一次，scheduler.step()是对lr进行调整，经过step_size次调用后更新一次lr（变为0.1倍）

    epoch_loss = total_loss / len(train_loader)
    print("epoch:{}  epoch_loss:{:.4f}".format(epoch, epoch_loss))
    if epoch % 10 == 0:
        save_checkpoint(model, params['save_name'], params['save_path'])


def val(best_cls_accuracy):

    print('===start eval===')
    model.eval()

    print("val from epoch: {}".format(epoch))
    progressbar = tqdm(range(len(val_loader)))
    total_cls_accuracy = 0.0
    with torch.no_grad():
        count = 0
        for batch_id, (img, class_label) in enumerate(val_loader):
            img, class_label = [x.cuda() for x in (img, class_label)]

            features, classes = model(img)

            cls_prob = F.softmax(classes, dim=-1)
            cls_pred = torch.argmax(cls_prob, dim=-1)
            cls_accuracy = np.mean((class_label == cls_pred).cpu().numpy())

            total_cls_accuracy += cls_accuracy

            progressbar.set_description("cls_accuracy: {:.4f} ".format(cls_accuracy))

            progressbar.update(1)
            count += 1
        progressbar.close()
        print('total_cls_accuracy: {:.4f} '.format(total_cls_accuracy))
        avg_cls_accuracy = total_cls_accuracy / count

        print("avg cls:{}".format(avg_cls_accuracy))

        if avg_cls_accuracy > best_cls_accuracy:
            best_cls_accuracy = avg_cls_accuracy
            save_checkpoint(model, params['save_name'], params['save_path'])

            print("current best_cls_accuracy: {:.4f} ".format(best_cls_accuracy))
        print(optimizer.state_dict()['param_groups'][0]['lr'])



def init_optimizer(model):
    ignored_params = list(map(id, model.auxiliary_module.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)
    return optimizer


def save_checkpoint(model, save_name, save_path):
    save_dict = {
        "epoch": epoch,
        "net": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_cls_accuracy": best_cls_accuracy
    }
    save_name = os.path.join(save_path, save_name + '.pth')
    torch.save(save_dict, save_name)
    print("model is saved: {}".format(save_name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--transfer", "-t", action="store_true")
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnext50'],
                        help='backbone network type')
    parser.add_argument('--gd_config', default='SG', type=str,
                        choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'],
                        help='global descriptors config')
    parser.add_argument('--feature_dim', default=1536, type=int, help='feature dim')
    parser.add_argument('--smoothing', default=0.1, type=float, help='smoothing value for label smoothing')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='temperature scaling used in softmax cross-entropy loss')
    parser.add_argument('--margin', default=0.1, type=float, help='margin of m for triplet loss')
    parser.add_argument('--save_name', default='CGD',type=str, help='the name of save model')
    parser.add_argument('--max_epoch', default=141, type=int, help='max trainning epoch')
    parser.add_argument('--num_class', "-n", default=719, type=int, help="the data classes" )
    parser.add_argument('--batchSize', "-b", default=64, type=int, help="trainning batch size")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with open('CGDparams.json') as f:
        params = json.load(f)
    args = parse_args()
    model = CBGModel(args.backbone_type, args.gd_config, args.feature_dim, args.num_class).cuda()
    model = nn.DataParallel(model, device_ids=device_ids)
    params_dict = {}
    # 这部分加载的是新的classifier层的参数
    if args.transfer:
        new_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        for key, value in new_dict.items():
            if 'auxiliary_module' in key:
                params_dict[key] = value
            else:
                pass

    class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=args.smoothing, temperature=args.temperature)
    feature_criterion = BatchHardTripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_lr'], gamma=0.1)
    best_cls_accuracy = 0.0

    print('====loading data====')
    load_time_start = time.time()
    train_loader = get_train_loader(lvDataset, args.batchSize)
    val_loader = get_val_loader(lvDataset, args.batchSize)
    load_time_end = time.time()
    print(load_time_end - load_time_start, ' s')
    print('====data loaded====')

    if args.resume:
        save_dict = torch.load(os.path.join(params['save_path'], 'CGD_mix.pth'))
        # 这部分加载除最后分类层以外的参数
        if args.transfer:
            for key, value in save_dict['net'].items():
                if 'auxiliary_module' in key:
                    pass
                else:
                    params_dict[key] = value
        else:
            params_dict = save_dict['net']
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(params_dict)
        else:
            model.load_state_dict(params_dict)

        #optimizer.load_state_dict(save_dict['optim'])
        #scheduler.load_state_dict(save_dict['scheduler'])
        current_epoch = save_dict['epoch'] + 1
        best_cls_accuracy = save_dict['best_cls_accuracy']
    else:
        current_epoch = 0
    print(current_epoch)
    for epoch in range(current_epoch, args.max_epoch):
        train()
        if epoch % 5 ==0:
            val(best_cls_accuracy)
