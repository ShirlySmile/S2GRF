import torch
from scipy import io
from torchstat import stat
from torch.nn import functional as F
from torch import nn, Tensor
from typing import Dict, Tuple

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        # if ch_out!=ch_in:
        self.extra = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
            nn.BatchNorm2d(ch_out)
        )

        self.prelu = nn.PReLU()
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        extra = self.extra(x)
        out = extra + out
        out = F.relu(out)
        # out = self.prelu(out)
        return out

class ChannelAttention(nn.Module):  
    def __init__(self, in_channel, r=0.3): 
        super(ChannelAttention, self).__init__()

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, in_channel),  
            nn.ReLU(True),
        )
        self.fc_siPool = nn.Sequential(
            nn.Linear(in_channel, in_channel), 
            nn.ReLU(True),
        )
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        # self.soft = nn.Softmax()

    def forward(self, x):

        avg_branch = self.AvgPool(x)
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        x1 = x.view(x.size(0), x.size(1), -1)
        Si = x1.std(2, unbiased=True)
        Si_weight = self.fc_siPool(Si)

        weight = avg_weight + Si_weight
        weight = self.soft(weight)

        h, w = weight.shape
        Mc = torch.reshape(weight, (h, w, 1, 1))

        x = Mc * x
        return x


class SpectralEncoder(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False):
        super(SpectralEncoder, self).__init__()

        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
                                      nn.ReLU(inplace=False))

        self.conv3d_2 = nn.Sequential(nn.Conv3d(1, 1, kernel_size=(5, 1, 1), stride=1, padding=(2, 0, 0)),
                                    nn.ReLU(inplace=False))
        self.pointwise_conv = nn.Sequential(nn.Conv2d(2 * ch_in, ch_out, 1, bias=bias),
                                            nn.BatchNorm2d(ch_out),
                                            nn.ReLU(inplace=False)
                                            )
        self.down1 = nn.MaxPool2d(2, 2)
        self.down2 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.bn2 = nn.BatchNorm2d(ch_in)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out1 = self.conv3d_1(x)
        out2 = self.conv3d_2(x)

        out1 = self.bn1(torch.squeeze(out1, 1))
        out2 = self.bn2(torch.squeeze(out2, 1))

        out1 = self.down1(out1)
        out2 = self.down2(out2)

        out = torch.cat([out1, out2], 1)
        out = self.pointwise_conv(out)
        return out





class IntraFusion_spectral(nn.Module):
    def __init__(self, ch_in1, ch_in2, ch_out, featuresize=16):
        super(IntraFusion_spectral, self).__init__()

        self.conv1x1 = nn.Sequential(nn.Conv2d(ch_in1 + ch_in2, ch_out, 1, bias=False),
                                    nn.BatchNorm2d(ch_out),
                                    nn.ReLU(inplace=False))


        self.conv1x1_1 = nn.Sequential(nn.Conv2d(ch_in1 + ch_in2, featuresize * featuresize, 1, bias=False),
                                    nn.BatchNorm2d(featuresize * featuresize),
                                    nn.ReLU(inplace=False))


        self.conv1x1_2 = nn.Sequential(nn.Conv2d(featuresize * featuresize, ch_in1 + ch_in2, 1, bias=False),
                                    nn.BatchNorm2d(ch_in1 + ch_in2),
                                    nn.ReLU(inplace=False))


        # normalization layer for the representations z1 and z2
        self.atten = ChannelAttention(ch_in1 + ch_in2)


    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        batch_size, in_channels, patch_h, patch_w = feature_map.shape
        # if in_channels != patch_h*patch_w:
        feature = self.conv1x1_1(feature_map)

        reshaped_fm = feature.reshape(batch_size, patch_h*patch_w, patch_h*patch_w)
        transposed_fm = reshaped_fm.transpose(1, 2)
        reshaped_fm = transposed_fm.reshape(batch_size, patch_h*patch_w, patch_h, patch_w)

        out_feature = self.conv1x1_2(reshaped_fm)

        return out_feature
    def forward(self, x11, x22):
        try:
            X = torch.cat([x11, x22], 1)
        except:
            X = x11
        X = self.unfolding(X)
        X = self.atten(X)

        out = self.conv1x1(X)

        c_out = out.view(out.size(0), out.size(1), -1)

        c = torch.matmul(c_out, c_out.transpose(1, 2))

        c_normalized = F.softmax(c, dim=1)
        c_normalized = F.softmax(c_normalized, dim=2)
        return out, c_normalized

class IntraFusion_spatial(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(IntraFusion_spatial, self).__init__()
        self.pointwise_conv = nn.Conv2d(ch_in*2, ch_out, 1, bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        self.atten = ChannelAttention(ch_in*2)

    def forward(self, x11, x22):
        X = torch.cat([x11, x22], 1)
        X = self.atten(X)


        out = self.pointwise_conv(X)
        out = F.relu(self.bn(out))

        c_out = out.view(out.size(0), out.size(1), -1)

        c = torch.matmul(c_out, c_out.transpose(1, 2))
        c_normalized = F.softmax(c, dim=1)
        c_normalized = F.softmax(c_normalized, dim=2)
        return out, c_normalized



class InterFusion(nn.Module):
    def __init__(self):
        super().__init__()
        attn_drop = 0.
        self.attn_drop1 = nn.Dropout(attn_drop)
        self.attn_drop2 = nn.Dropout(attn_drop)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)


    def forward(self, x1, x2):

        B, C, H, W = x1.shape
        x111 = x1.reshape(B, C, -1)
        x211 = x2.reshape(B, C, -1)

        x11 = F.normalize(x111, p=2, dim=2, eps=1e-12)
        x21 = F.normalize(x211, p=2, dim=2, eps=1e-12)


        attn1 = (x11 @ x11.transpose(-2, -1))
        attn2 = (x21 @ x21.transpose(-2, -1))
        attn0 = (x21 @ x11.transpose(-2, -1))
        attn00 = (x11 @ x21.transpose(-2, -1))
        #
        # att000 = attn0.transpose(-2, -1)

        attn11 = attn1+attn0
        attn11 = self.softmax1(attn11)
        attn11 = self.attn_drop1(attn11)


        attn22 = attn2+attn00
        attn22 = self.softmax2(attn22)
        attn22 = self.attn_drop2(attn22)

        x1_o = (attn11 @ x111).transpose(1, 2).reshape(B, C, H, W) + x1
        x2_o = (attn22 @ x211).transpose(1, 2).reshape(B, C, H, W) + x2

        x_out = x1_o + x2_o
        return x_out


class SpatialConv(nn.Module):
    def __init__(self, ch_in, stride=1):
        super(SpatialConv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x, _ = torch.max(x, dim=1, keepdim=True)
        return x

class Spatialencoder(nn.Module):
    def __init__(self,ch_in, ch_out, stride=1):
        super(Spatialencoder,self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.conv2 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv3 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, groups=ch_out)
        self.bn3 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(ch_out)
        )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.conv3(out)
        extra = self.extra(x)
        out = extra + out
        out = F.relu(out)
        return out



class S2GRF(nn.Module):
    def __init__(self, in_channels1, in_channels2, num_classes, ratio=4):
        super(S2GRF,self).__init__()

        self.ratio = ratio
        self.Spectralencoder1_1 = SpectralEncoder(in_channels1, 64)
        self.Spectralencoder2_1 = SpectralEncoder(in_channels2, 8)

        self.IntraFusion_spectral = IntraFusion_spectral(64, 8, 64)

        self.RP0 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )
        self.Resblk1 = ResBlk(128, 128)
        self.Resblk2 = ResBlk(128, 64)

        self.Spatialencoder3_1 = ResBlk(in_channels1, 64, stride=2)
        self.Spatialencoder4_1 = ResBlk(in_channels2, 64, stride=2)

        self.IntraFusion_spatial = IntraFusion_spatial(64, 64)

        self.RP = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )
        self.Resblk3 = ResBlk(128, 128)
        self.Resblk4 = ResBlk(128, 64)

        self.InterFusion = InterFusion()
        self.Resblk5 = ResBlk(128, 128)

        self.RP1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )

        self.RP2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        )


        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )

    def forward(self, HS, MS):
        branch1 = self.Spectralencoder1_1(HS)
        branch2 = self.Spectralencoder2_1(MS)


        path1, c1 = self.IntraFusion_spectral(branch1, branch2)

        # branch3 = self.Spatialencoder3_1(HS)
        # branch4 = self.Spatialencoder4_1(MS)
        #
        #
        # path2, c2 = self.IntraFusion_spatial(branch3, branch4)



        path1 = self.RP0(path1)
        # path2 = self.RP(path2)



        # out_blk = self.InterFusion(path1, path2)
        out_blk = self.Resblk5(path1)
        out_blk = self.RP2(out_blk)
        input_FC = out_blk.contiguous().view(out_blk.shape[0], -1)
        out_FC = self.classifier(input_FC)

        return out_FC, c1


class MRloss(nn.Module):
    def __init__(self):
        super(MRloss, self).__init__()
        self.beta = 1e-3

    def off_diagonal(self, x):
        c, n, m = x.shape
        assert n == m
        x1 = x.view(c, -1)
        t = x1[:, :-1]
        t1 = t.view(c, n - 1, n + 1)
        t2 = t1[:, :, 1:]
        t3 = t2.flatten()
        return t3

    def forward(self, x, label):
        o1, c1 = x
        on_diag1 = (torch.diagonal(c1)-1).pow(2).mean()
        off_diag1 = torch.norm(self.off_diagonal(c1), p=2)

        info_loss1 = on_diag1 + off_diag1


        loss = info_loss1
        return loss


if __name__ == "__main__":

    HS = torch.randn(5, 4, 32, 32)
    MS = torch.randn(5, 1, 32, 32)
    y = torch.randint(0, 7, (1, 5))

    grf_net = S2GRF(in_channels1=4, in_channels2=1, num_classes=11)
    grf_net.cuda()
    for m in grf_net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    out_result = grf_net(HS.cuda(), MS.cuda())

    LossFunc = MRloss()
    print(out_result)
    stat(grf_net, [(4, 32, 32), (1, 32, 32)], device="cuda")

  