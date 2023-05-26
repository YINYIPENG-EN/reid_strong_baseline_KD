import torchsnooper

from .resnet import conv1x1, conv3x3, BasicBlock, Bottleneck
import torch.nn as nn


def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )


class SepConv(nn.Module):
    '''
    这个卷积名字起得花里胡哨的，其实总结起来就是输入通道每个通道一个卷积得到和输入通道数相同的特征图，然后再使用若干个1*1的卷积聚合每个特征图的值得到输出特征图。
    假设我们输入通道是16，输出特征图是32，并且使用3*3的卷积提取特征，那么第一步一共需要16*3*3个参数，第二步需要32*16*1*1个参数，一共需要16*3*3+32*16*1*1=656个参数。
    '''

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        # 3*3 s=2 p=1
        # 1*1 s=1 p=1
        # BN
        # Relu
        # 3*3 s=1 p=1
        # 1*1 s=1 p=0
        # BN
        # Relu
        self.op = nn.Sequential(
            # 分组卷积，这里的分组数=输入通道数，那么每个group=channel_in/channel_in=1个通道，就是每个通道进行一个卷积
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            # affine 设为 True 时，BatchNorm 层才会学习参数 gamma 和 beta，否则不包含这两个变量，变量名是 weight 和 bias。
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            # 分组卷积
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        '''
        x-->conv_3x3_s2(分组卷积)-->conv_1x1-->bn-->relu-->conv_3x3(分组卷积)-->conv_1x1-->bn-->relu-->out
        '''
        return self.op(x)


def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )


class ResNet_KD(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_stride=2):
        super(ResNet_KD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        # 空洞卷积定义
        self.dilation = 1
        # 是否用空洞卷积代替步长，如果不采用空洞卷积，均为False
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups  # 分组卷积分组数
        self.base_width = width_per_group  # 卷积宽度

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # bn层
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # 尺寸不变
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # 尺寸减半
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])  # 尺寸减半
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])
        '''
        此处和原Resnet不同，原Resnet这里是自适应平均池化，然后接一个全连接层。
        scala层的作用是对特征层的H，W做缩放处理，因为要和深层网络中其他Bottleneck输出特征层之间做loss
        '''
        self.scala1 = nn.Sequential(
            # 输入通道64*4=256，输出通道128*4=512
            SepConv(  # 尺寸减半
                channel_in=64 * block.expansion,
                channel_out=128 * block.expansion
            ),
            # 输入通道128*4=512， 输出通道256*4=1024
            SepConv(  # 尺寸减半
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion
            ),
            # 输入通道256*4=1024，输出通道512*4=2048
            SepConv(  # 尺寸减半
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion
            ),
            # 平均池化
            nn.AvgPool2d(4, 4)
        )
        self.scala2 = nn.Sequential(
            # 输入通道128*4=512，输出通道1024
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=256 * block.expansion,
            ),
            # 输入通道256*4=1024，输出通道512*4=2048
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # 平均池化
            nn.AvgPool2d(4, 4)
        )
        self.scala3 = nn.Sequential(
            # 输入通道256*4=1024，输出通道512*4=2048
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=512 * block.expansion,
            ),
            # 平均池化
            nn.AvgPool2d(4, 4)
        )
        # 平均池化
        #self.scala4 = nn.AvgPool2d(4, 4)
        self.scala4 = nn.AdaptiveAvgPool2d(output_size=[2,1])

        self.attention1 = nn.Sequential(
            SepConv(  # 尺寸减半
                channel_in=64 * block.expansion,  # 256
                channel_out=64 * block.expansion  # 256
            ),  # 比输入前大两个像素
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),  # 恢复原来尺寸
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=128 * block.expansion,
                channel_out=128 * block.expansion
            ),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention3 = nn.Sequential(
            SepConv(
                channel_in=256 * block.expansion,
                channel_out=256 * block.expansion
            ),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 残差边采用1x1卷积升维条件，即当步长不为1或者输入通道数不等于输出通道数的时候
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # layers用来存储每个当前残差层的所有残差块
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        # 仅在第一个bottleneck采用1x1进行升维，其他的bottleneck是直接输入和输出相加
        return nn.Sequential(*layers)

    def forward(self, x):
        # 以x = (batch_size,3,256,128)为例
        feature_list = []
        x = self.conv1(x)  # get batch_size,64,128,64
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # conv2_x  输出256通道  batch_size,256,64,32

        fea1 = self.attention1(x)  # 输出通道为256 shape[batch_size,256,64,32]
        fea1 = fea1 * x
        feature_list.append(fea1)  # feature_list:[(batch, 256, 64, 32)]

        x = self.layer2(x)  # conv3_x  (batch_size, 512, 32, 16)

        fea2 = self.attention2(x)  # (batch_size, 512, 32, 16)
        fea2 = fea2 * x
        feature_list.append(fea2)  # feature_list:[(batch_size, 256, 64, 32),(batch_size, 512, 32, 16)]

        x = self.layer3(x)  # conv4_x (batch_size, 1024, 16, 8)

        fea3 = self.attention3(x)  # (batch_size, 1024, 16, 8)
        fea3 = fea3 * x
        feature_list.append(fea3)  # feature_list:[(batch_size, 256, 64, 32),(batch_size, 512, 32, 16),(batch_size, 1024, 16, 8)]

        x = self.layer4(x)  # conv5_x  最深层网络 batch_size,2048,16,8
        feature_list.append(x)  # feature_list:[(batch_size, 256, 64, 32),(batch_size, 512, 32, 16),(batch_size, 1024, 16, 8),(batch_size,2048,16,8)]

        # feature_list[0].shape is [batch_size, 256, 64, 32], scala1 shape is [batch,2048,2,1] view is [batch,2048*2]
        out1_feature = self.scala1(feature_list[0]).view(x.size(0), -1)  # # 得到新的特征图 对应到论文中的Bottleneck1
        # feature_list[1].shape is [batch_size, 512, 32, 16], scala2 shape is [batch,2048,2,1] view is [batch,2048*2]
        out2_feature = self.scala2(feature_list[1]).view(x.size(0), -1)  # 得到新的特征图 对应到论文中的Bottleneck2
        # feature_list[2].shape is [batch_size, 1024, 16, 8],scala3 shape is [batch,2048,2,1] view is [batch,2048*2]
        out3_feature = self.scala3(feature_list[2]).view(x.size(0), -1)  # 得到新的特征图 对应到论文中的Bottleneck3
        # feature_list[3].shape is [batch_size,2048,16,8],scala4 shape is [1,2048,7,7], view is [1,2048*7*7]
        out4_feature = self.scala4(feature_list[3]).view(x.size(0), -1)  # conv5_x  最深层网络

        # out1 = self.fc1(out1_feature)
        # out2 = self.fc2(out2_feature)
        # out3 = self.fc3(out3_feature)
        # out4 = self.fc4(out4_feature)
        # 返回的特征层分别是经过全连接和不仅过全连接的
        # return [out4, out3, out2, out1], [out4_feature, out3_feature, out2_feature, out1_feature]
        return feature_list, [out4_feature, out3_feature, out2_feature, out1_feature]
