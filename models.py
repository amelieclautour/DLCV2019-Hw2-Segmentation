import torch
import torch.nn as nn
import torchvision.models as models


class Net_improved(nn.Module):

    def __init__(self, args):
        super(Net_improved, self).__init__()


        ''' declare layers used in this network'''
        resnet34 = models.resnet34(pretrained=True)
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.l1 = resnet34.layer1
        self.l2 = resnet34.layer2
        self.l3 = resnet34.layer3
        self.l4 = resnet34.layer4


        # self.trans_conv11 = nn.ConvTranspose2d(2048, 1024, 4, 1, 1, bias=False)  # 11x14 -> 12x13
        # self.relu11 = nn.ReLU()
        #
        # self.trans_conv12 = nn.ConvTranspose2d(1024, 512, 4, 1, 2, bias=False)  # 12x13 -> 11x14
        # self.relu12 = nn.ReLU()

        # first block
        self.trans_conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False) #11x14 -> 22x28
        self.relu1 = nn.ReLU()

        # second block

        self.trans_conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1 ,bias=False) #22x28->44x56
        self.relu2 = nn.ReLU()

        # third block
        self.trans_conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False) #44x56 ->88x112
        self.relu3 = nn.ReLU()

        # fourth block
        self.trans_conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.relu4 = nn.ReLU()

        # fifth block
        self.trans_conv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False)
        self.relu5 = nn.ReLU()

        # sixth block
        self.conv = nn.Conv2d(16, 9, 1, 1, 0, bias=True)


    def forward(self, img):

        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = self.trans_conv1(x)
        x = self.relu1(x)

        x = self.trans_conv2(x)
        x = self.relu2(x)

        x = self.trans_conv3(x)
        x = self.relu3(x)

        x = self.trans_conv4(x)
        x = self.relu4(x)

        x = self.trans_conv5(x)
        x = self.relu5(x)

        return x


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()


        ''' declare layers used in this network'''
        resnet18 = models.resnet18(pretrained=True)
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.l1 = resnet18.layer1
        self.l2 = resnet18.layer2
        self.l3 = resnet18.layer3
        self.l4 = resnet18.layer4
        #self.fc = resnet18.fc
        #self.avgpool = resnet18.avgpool

        # first block
        self.trans_conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False) #11x14 -> 22x28
        self.relu1 = nn.ReLU()

        # second block

        self.trans_conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1 ,bias=False) #22x28->44x56
        self.relu2 = nn.ReLU()

        # third block
        self.trans_conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False) #44x56 ->88x112
        self.relu3 = nn.ReLU()

        # fourth block
        self.trans_conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.relu4 = nn.ReLU()

        # fifth block
        self.trans_conv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False)
        self.relu5 = nn.ReLU()

        # sixth block
        self.conv = nn.Conv2d(16, 9, 1, 1, 0, bias=True)


    def forward(self, img):
        #resnet18
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        #x = self.avgpool(x)
        #x = self.fc(x)

        x = self.trans_conv1(x)
        x = self.relu1(x)

        x = self.trans_conv2(x)
        x = self.relu2(x)

        x = self.trans_conv3(x)
        x = self.relu3(x)

        x = self.trans_conv4(x)
        x = self.relu4(x)

        x = self.trans_conv5(x)
        x = self.relu5(x)

        return x
