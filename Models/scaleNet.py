from torch import nn
from Models.DenseNet import densenet121, densenet161
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from collections import OrderedDict
from Datasets.lvDataset import lvDataset
from torch.utils.data import DataLoader


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, ):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        temp = out
        out = self.classifier(out)
        return out, temp


class scaleNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=252, ):

        super(scaleNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config[:-1]):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2

        # 非共享的全局表达
        self.block_global = _DenseBlock(num_layers=block_config[-1], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.block_local = _DenseBlock(num_layers=block_config[-1], num_input_features=num_features,
                                        bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[-1] * growth_rate
        # Final batch norm
        self.batch_norm = nn.BatchNorm2d(num_features)
        #self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier_global = nn.Linear(num_features, num_classes)
        self.classifier_delocal = nn.Linear(num_features, num_features//8)
        self.classifier_local_0 = nn.Linear(num_features // 8, num_classes)
        self.classifier_local_1 = nn.Linear(num_features // 8, num_classes)
        self.classifier_local_2 = nn.Linear(num_features // 8, num_classes)
        self.classifier_local_3 = nn.Linear(num_features // 8, num_classes)
        self.classifier_local_center = nn.Linear(num_features // 8, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x_global):
        feature_embeddings = []
        class_embeddings = []
        features_global = self.features(x_global)
        block4_global = self.block_global(features_global)
        norm_global = self.batch_norm(block4_global)
        out_global = F.relu(norm_global, inplace=True)
        out_global = F.adaptive_avg_pool2d(out_global, (1, 1)).view(out_global.size(0), -1)
        feature_embeddings_global = out_global
        class_global = self.classifier_global(feature_embeddings_global)

        feature_embeddings.append(feature_embeddings_global)
        class_embeddings.append(class_global)

        block4_local = self.block_local(features_global)
        normal_local = self.batch_norm(block4_local)
        out_local = F.relu(normal_local, inplace=True)
        out_local_4 = F.adaptive_avg_pool2d(out_local, (2, 2)).view((out_local.size(0), -1, 4)) # 4=2*2 将特征图分为四块
        out_local_center = F.adaptive_avg_pool2d(out_local, (4, 4)).view(out_local.size(0), -1, 16)

        local_center = torch.cat((out_local_center[:, :, 5], out_local_center[:, :, 6], out_local_center[:, :, 9], out_local_center[:, :,10]),
                                 dim=-1).view((out_local_center.size(0), -1, 2, 2))

        local_center = F.adaptive_avg_pool2d(local_center, (1, 1)).view((local_center.size(0), -1))

        local_part_0 = out_local_4[:, :, 0]
        local_part_1 = out_local_4[:, :, 1]
        local_part_2 = out_local_4[:, :, 2]
        local_part_3 = out_local_4[:, :, 3]

        local_part_embeddings_0 = self.classifier_delocal(local_part_0)
        feature_embeddings.append(local_part_embeddings_0)

        local_part_embeddings_1 = self.classifier_delocal(local_part_1)
        feature_embeddings.append(local_part_embeddings_1)

        local_part_embeddings_2 = self.classifier_delocal(local_part_2)
        feature_embeddings.append(local_part_embeddings_2)

        local_part_embeddings_3 = self.classifier_delocal(local_part_3)
        feature_embeddings.append(local_part_embeddings_3)

        local_center_embeddings =  self.classifier_delocal(local_center)
        feature_embeddings.append(local_center_embeddings)

        class_local_0 = self.classifier_local_0(feature_embeddings[1])
        class_local_1 = self.classifier_local_1(feature_embeddings[2])
        class_local_2 = self.classifier_local_2(feature_embeddings[3])
        class_local_3 = self.classifier_local_3(feature_embeddings[4])
        class_local_center = self.classifier_local_center(feature_embeddings[-1])

        class_embeddings.append(class_local_0)
        class_embeddings.append(class_local_1)
        class_embeddings.append(class_local_2)
        class_embeddings.append(class_local_3)
        class_embeddings.append(class_local_center)

        return feature_embeddings, class_embeddings


if __name__ == "__main__":
    device = torch.device('cuda:0')
    net = scaleNet()
    net.to(device)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train_global = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),  # 训练集才需要做这一步处理，获得更多的随机化数据
        transforms.ToTensor(),
        normalize])


    dataset = lvDataset(train=True, transform_train=transform_train_global)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch_id, (img_globals, labels) in enumerate(train_loader):
        img_globals, labels = \
            [x.cuda(0) for x in (img_globals, labels)]
        features, y = net(img_globals)
        entropy_loss_global = nn.CrossEntropyLoss()(y[0], labels)

        break