import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 32, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    if model == 'ConvNetBN':
        net = ConvNet(channel=channel,
                      num_classes=num_classes,
                      net_width=net_width,
                      net_depth=net_depth,
                      net_act=net_act,
                      net_norm='batchnorm',
                      net_pooling=net_pooling,
                      im_size=im_size)
    elif model == 'ConvNet_1D':
        net = ConvNet_1D(channel=channel,
                         num_classes=num_classes,
                         net_width=256,
                         net_depth=net_depth,
                         net_act=net_act,
                         net_norm='batchnorm',
                         net_pooling='avgpooling')
    elif model == 'ConvNet_1D_Bi':
        net = ConvNet_1D_Bi(channel=channel,
                         num_classes=num_classes,
                         net_width=256,
                         net_depth=net_depth,
                         net_act=net_act,
                         net_norm='batchnorm',
                         net_pooling='avgpooling')
    elif model == 'Binary_Classifier':
        net = Binary_Classifier(input_size=channel, num_classes=num_classes)
    elif model == 'ResNet50':
        net = ResNet50(channel=channel, num_classes=num_classes)
    elif model == 'ResNet110':
        net = ResNet110(channel=channel, num_classes=num_classes)

    else:
        net = None
        exit('unknown model: %s' % model)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = 'cuda'
        if gpu_num > 1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net





# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


class Binary_Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Binary_Classifier, self).__init__()
        # self.fc1 = nn.Linear(input_size, 64)  # 输入层到隐藏层
        # self.fc2 = nn.Linear(64, 32)  # 隐藏层
        # self.fc3 = nn.Linear(32, num_classes, bias=False)  # 输出层，输出1个值用于二分类
        self.fc1 = nn.Linear(input_size, num_classes, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = torch.relu(self.fc1(x))  # 隐藏层激活
        # x = torch.relu(self.fc2(x))  # 隐藏层激活
        # x = self.fc3(x)              # 输出层
        x = self.fc1(x)
        # x = self.sigmoid(x)          # Sigmoid激活，输出概率（0-1）
        return x




class ConvNet_1D(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling):
        super(ConvNet_1D, self).__init__()

        self.features, self.num_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling)
        # self.classifier = nn.Linear(self.num_feat, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

    def clf(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = x.view(x.size(0), -1)
        # out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.contiguous().view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool1d(kernel_size=1, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool1d(kernel_size=1, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm1d(shape_feat, affine=True, track_running_stats=True)
        elif net_norm == 'layernorm':
            return nn.LazyBatchNorm1d(shape_feat, affine=True)   # NLP or small batch-size
        elif net_norm == 'instancenorm':
            return nn.InstanceNorm1d(shape_feat, affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling):
        layers = []
        in_channels = channel

        for d in range(net_depth):
            layers += [nn.Conv1d(in_channels, net_width, kernel_size=1, stride=2)]
            in_channels = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, in_channels)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                # in_channels = int(in_channels / 2)
                net_width = int(net_width / 2)
        return nn.Sequential(*layers), in_channels


class ConvNet_1D_Bi(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling):
        super(ConvNet_1D_Bi, self).__init__()

        self.features, self.num_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling)
        self.classifier = nn.Linear(self.num_feat, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.sigmoid(out)
        return out

    def clf(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = self.sigmoid(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.contiguous().view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool1d(kernel_size=1, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool1d(kernel_size=1, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm1d(shape_feat, affine=True, track_running_stats=True)
        elif net_norm == 'layernorm':
            return nn.LazyBatchNorm1d(shape_feat, affine=True)   # NLP or small batch-size
        elif net_norm == 'instancenorm':
            return nn.InstanceNorm1d(shape_feat, affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling):
        layers = []
        in_channels = channel

        for d in range(net_depth):
            layers += [nn.Conv1d(in_channels, net_width, kernel_size=1, stride=2)]
            in_channels = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, in_channels)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                # in_channels = int(in_channels / 2)
                net_width = int(net_width / 2)
        return nn.Sequential(*layers), in_channels


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.contiguous().view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


class ResNet(nn.Module):

    def __init__(self, depth, channels, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'bottleneck':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = Bottleneck
        elif block_name.lower() == 'basicblock':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = BasicBlock
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def ResNet50(channel, num_classes):
    return ResNet(50, channels=channel, num_classes=num_classes, block_name='BasicBlock')

def ResNet110(channel, num_classes):
    return ResNet(110, channels=channel, num_classes=num_classes, block_name='Bottleneck')