from torch import nn
from utils.fmodule import FModule
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.mobile_net = mobilenet_v2(MobileNet_V2_Weights.IMAGENET1K_V2)
        self.mobile_net.classifier[1] = nn.Linear(1280, 100, bias=True)

    def forward(self, x):
        x = self.mobile_net(x)
        return x


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)