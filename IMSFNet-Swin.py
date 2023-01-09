import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet50 import ResNet50
import torchvision.models as models
from swin_encoder import SwinTransformer
class SPRM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPRM, self).__init__()
        self.convert =nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.branch1 = nn.Sequential(
             nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=8, dilation=8),nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel*5, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.shortconnect = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x0 = self.convert(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(torch.add(x0,x1))
        x3 = self.branch3(torch.add(torch.add(x0,x1),x2))
        x4 = self.branch4(torch.add(torch.add(torch.add(x0,x1),x2),x3))
        x5 = self.branch5(torch.add(torch.add(torch.add(torch.add(x0,x1),x2),x3),x4))
        x_res1 = self.shortconnect(x0)
        x_cat = torch.cat((x1, x2, x3, x4,x5),1)
        x_ = self.reduce(x_cat)
        x = self.relu(x_+x_res1)
        return x

class GFEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GFEM, self).__init__()
        self.convert =nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 3), padding=(0, 1), dilation=1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, (3, 1), padding=(1, 0), dilation=1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3),nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1,5), padding=(0,2), dilation=1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, (5,1), padding=(2,0), dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5), nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 7), padding=(0,3), dilation=1),nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, (7, 1), padding=(3,0), dilation=1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.branch6 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.branch7 = nn.Sequential(

            nn.Conv2d(out_channel, out_channel, (1, 9), padding=(0,4), dilation=1),nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, (9, 1), padding=(4,0), dilation=1),nn.BatchNorm2d(out_channel), nn.ReLU(True),

        )
        self.branch8 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=9, dilation=9), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel*5, out_channel, 1), nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
        self.relu = nn.ReLU(True)
        self.short = nn.Sequential(
            nn.Conv2d(out_channel,out_channel,1),nn.BatchNorm2d(out_channel),nn.ReLU(True),
        )
    def forward(self, x):
        x0 = self.convert(x)
        x1_ = self.branch1(x0)
        x1 = self.branch2(x1_)
        x2_ = self.branch3(torch.add(x1_, x0))
        x2 = self.branch4(torch.add(x1,x2_))
        x3_ = self.branch5(torch.add(x2_, x0))
        x3 = self.branch6(torch.add(x2,x3_))
        x4_ = self.branch7(torch.add(x3_, x0))
        x4 = self.branch8(torch.add(x3,x4_))
        y = torch.cat((x0, x1, x2, x3, x4),1)
        x_cat = self.reduce(y)
        x_res = self.short(x0)
        x = self.relu(x_cat+x_res)
        return x

class SE(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SPRNet(nn.Module):#整个模型
    def __init__(self,channel=192):
        super(SPRNet, self).__init__()
        self.resnet = ResNet50()
        self.encoder = SwinTransformer(img_size=384,
                                       embed_dim=128,
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       window_size=12)

        pretrained_dict = torch.load('/mnt/CFPNet/method_pth/swin_base_patch4_window12_384_22k.pth')["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)



        self.sprm1 = SPRM(1024,channel)#统一降维成64
        self.sprm2 = SPRM(512,channel)
        self.sprm3 = SPRM(256,channel)
        self.sprm4 = SPRM(128,channel)
        #self.sprm5 = SPRM(64,channel)
        self.gfem  = GFEM(1024,channel)
        self.se = SE(channel,channel)
        self.reduce = nn.Sequential(
            nn.Conv2d(channel,1,1)
        )
        self.reduce1 = nn.Sequential(
            nn.Conv2d(channel*3,channel,1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )
        self.reduce2 = nn.Sequential(
            nn.Conv2d(channel*4,channel,1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )
        self.reduce3 = nn.Sequential(
            nn.Conv2d(channel*2,channel,1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )
        self.reduce4 = nn.Sequential(
            nn.Conv2d(channel*5,channel,1),nn.BatchNorm2d(channel),nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):#整个模型参数
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        #self.initialize_weights()

    def forward(self, x):
        features = self.encoder(x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        if len(features) > 4:
            x5 = features[4]
        x_size = x.size()[2:]
        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]
        x5_size = x5.size()[2:]

        y6 = self.gfem(x2)
        f6_= y6
        f6 = self.reduce(f6_)
        score6 = F.interpolate(f6,x_size,mode='bilinear',align_corners=True)
        y6_5 = F.interpolate(y6,x2_size,mode='bilinear',align_corners=True)
        y5 = self.sprm1(x2)
        f5_= torch.cat((y6_5,y5),1)
        f5_ = self.reduce3(f5_)
        f5_= torch.add(y6_5,f5_)+self.se(y5)
        f5_= torch.add(y5,f5_)

        f5 = self.reduce(f5_)
        score5 = F.interpolate(f5, x_size, mode='bilinear', align_corners=True)#上采样过程
        f5_4 = F.interpolate(f5_,x3_size,mode='bilinear',align_corners=True)
        y4 = self.sprm2(x3)
        f4_= torch.cat((f5_4,y4),1)
        f4_=self.reduce3(f4_)
        f4_=torch.add(f5_4,f4_)+self.se(y4)
        f4_=torch.add(y4,f4_)

        f4 = self.reduce(f4_)
        score4 = F.interpolate(f4,x_size,mode='bilinear',align_corners=True)
        f4_3 = F.interpolate(f4_,x4_size,mode='bilinear',align_corners=True)
        y3 = self.sprm3(x4)
        f3_= torch.cat((f4_3,y3),1)
        f3_= self.reduce3(f3_)
        f3_= torch.add(f4_3,f3_)+self.se(y3)
        f3_= torch.add(y3,f3_)
        f3 = self.reduce(f3_)
        score3 = F.interpolate(f3,x_size,mode='bilinear',align_corners=True)
        f3_2 = F.interpolate(f3_,x5_size,mode='bilinear',align_corners=True)
        y2 = self.sprm4(x5)
        f2_= torch.cat((f3_2,y2),1)
        f2_= self.reduce3(f2_)
        f2_= torch.add(f3_2,f2_)+self.se(y2)
        f2_= torch.add(y2,f2_)
        f2 = self.reduce(f2_)
        score2 = F.interpolate(f2,x_size,mode='bilinear',align_corners=True)

        return score2,score3,score4,score5,score6

    """
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        self.resnet.load_state_dict(res50.state_dict(), False)
    """

if __name__ == '__main__':
    a = SPRNet()
    print(a)