import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -- Model Factory --
def _load_pretrained(model, url, num_classes):
    state_dict = model_zoo.load_url(url)
    if num_classes != 1000:
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
    model.load_state_dict(state_dict, strict=False)
    return model

def resnet18(pretrained=False, num_classes=1, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
    return _load_pretrained(model, model_urls["resnet18"], num_classes) if pretrained else model

def resnet34(pretrained=False, num_classes=1, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return _load_pretrained(model, model_urls["resnet34"], num_classes) if pretrained else model

def resnet50(pretrained=False, num_classes=1, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return _load_pretrained(model, model_urls["resnet50"], num_classes) if pretrained else model

def resnet101(pretrained=False, num_classes=1, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    return _load_pretrained(model, model_urls["resnet101"], num_classes) if pretrained else model

def resnet152(pretrained=False, num_classes=1, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
    return _load_pretrained(model, model_urls["resnet152"], num_classes) if pretrained else model
