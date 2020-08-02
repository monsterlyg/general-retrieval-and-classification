import json
import os
import sys
import time
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
device_ids = [0, 2]

import torch.nn.functional as F
from torch import optim, nn
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from torch.optim import lr_scheduler
from tqdm import tqdm
from Datasets.lvDataset import lvDataset
from Models.scaleNet import scaleNet
from loss import compute_feature_similarity_loss

import torch

criterion = nn.CrossEntropyLoss()


def get_train_loader(DataSet, batchSize):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train_global = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=180),
        transforms.ToTensor(),
        normalize])


    train_set = DataSet(train=True, transform_train=transform_train_global)
    train_loader = torchdata.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=2,
                                        pin_memory=False, drop_last=True)

    return train_loader


def train(net, optimizer, scheduler, params, start_epoch):
    print('====loading data====')
    load_time_start = time.time()
    train_loader = get_train_loader(lvDataset, params['batchSize'])
    load_time_end = time.time()
    print(load_time_end - load_time_start, ' s')
    print('====data loaded====')

    #device = torch.device(params['device'])
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    #net.to(device)
    print('===start training===')
    target = torch.ones(128).cuda()  # 局部特征表达的维数
    for epoch in range(start_epoch, params['MAX_EPOCHES']):
        print("Train Epoch: {}".format(epoch))
        total_loss = 0
        net.train()
        progressbar = tqdm(range(len(train_loader)))
        for batch_id, (img_globals, labels) in enumerate(train_loader):
            img_globals, labels = \
                [x.cuda() for x in (img_globals, labels)]
            # net.zero_grad()
            optimizer.zero_grad()
            if img_globals.shape[0] < params['batchSize']:
                continue

            features, y = net(img_globals)
            y_global, y1, y2, y3, y4, y_center = y
            entropy_loss_global = criterion(y_global, labels)
            entropy_loss_local_0 = criterion(y1, labels)
            entropy_loss_local_1 = criterion(y2, labels)
            entropy_loss_local_2 = criterion(y3, labels)
            entropy_loss_local_3 = criterion(y4, labels)
            entropy_loss_local_center = criterion(y_center, labels)
            entropy_loss = entropy_loss_global + 0.5*(entropy_loss_local_0 + entropy_loss_local_1 +
                                              entropy_loss_local_2 + entropy_loss_local_3 + entropy_loss_local_center)

            similarity_loss = 50 * compute_feature_similarity_loss(features, target)
            loss = entropy_loss + similarity_loss
            loss.backward()

            optimizer.step()

            total_loss += loss
            progressbar.set_description("entropy loss: {:.4f}   simialrity loss:{:.4f}".
                                        format(entropy_loss.item(), similarity_loss.item()))
            progressbar.update(1)
        progressbar.close()

        scheduler.step()  # 每个epoch里用一次，scheduler.step()是对lr进行调整，经过step_size次调用后更新一次lr（变为0.1倍）

        epoch_loss = total_loss.item() / len(train_loader)
        print(f'epoch: {epoch:03d} epoch_loss: {epoch_loss:.4f}', end='\n')

        if epoch % 20 == 0 or epoch == params['MAX_EPOCHES'] - 1:
            save_dict = {
                "epoch": epoch,
                "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            save_name = os.path.join(params['save_path'], params['save_name'] + '.pth')
            torch.save(save_dict, save_name)
            print("model is saved: {}".format(save_name))
            print(optimizer.state_dict()['param_groups'][0]['lr'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--transfer", "-t", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with open('params.json') as f:
        params = json.load(f)
    args = parse_args()
    net = scaleNet().cuda()

    optimizer = optim.Adam(net.parameters(), lr=params['lr'])
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_lr'], gamma=0.1)

    if args.resume:
        save_dict = torch.load(os.path.join(params['save_path'], params['save_name'] + '.pth'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        exp_lr_scheduler.load_state_dict(save_dict['scheduler'])
        start_epoch = save_dict['epoch'] + 1
    else:
        start_epoch = 0
    train(net, optimizer, exp_lr_scheduler, params, start_epoch)
