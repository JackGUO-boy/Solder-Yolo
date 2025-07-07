import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
__all__ = ['GAM', 'CBAM', 'CoordAtt', 'ECA','SolderCbam','HCAtt','HCSAtt','CoordAtt_HOnly','HCAtt_Max','H_MAx_C_MaxAtt']


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class GAM(nn.Module):
    def __init__(self, in_channels, rate=4):
        super().__init__()
        out_channels = in_channels  #输入特征图的通道数。
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        inchannel_rate = int(in_channels / rate)

        self.linear1 = nn.Linear(in_channels, inchannel_rate) #两个全连接层，用于计算通道注意力
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(inchannel_rate, in_channels) #self.linear1 将输入的通道数压缩为 in_channels / rate，然后 self.linear2 将其恢复到原来的通道数

        self.conv1 = nn.Conv2d(in_channels, inchannel_rate, kernel_size=7, padding=3, padding_mode='replicate')
        #两个卷积层用于生成空间注意力。卷积核大小为7，填充为3（保持输出特征图的空间大小不变）
        self.conv2 = nn.Conv2d(inchannel_rate, out_channels, kernel_size=7, padding=3, padding_mode='replicate')

        self.norm1 = nn.BatchNorm2d(inchannel_rate) #批量归一化层，分别用于 conv1 和 conv2 的输出
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # B,C,H,W ==> B,H*W,C
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c) # 将输入特征图从 (b, c, h, w) 转置并重塑为 (b, h*w, c)，即将空间维度展平，得到每个像素位置的特征

        # B,H*W,C ==> B,H,W,C
        x_att_permute = self.linear2(self.relu(self.linear1(x_permute))).view(b, h, w, c) #计算通道注意力

        # B,H,W,C ==> B,C,H,W
        x_channel_att = x_att_permute.permute(0, 3, 1, 2) # 将通道注意力的维度重新排列回 (b, c, h, w)。

        x = x * x_channel_att # 将通道注意力应用到原始输入特征图 x 上

        x_spatial_att = self.relu(self.norm1(self.conv1(x)))
        x_spatial_att = self.sigmoid(self.norm2(self.conv2(x_spatial_att)))

        out = x * x_spatial_att

        return out


class ChannelAttention(nn.Module):  # 通道注意力模块
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) #对输入的特征图进行全局平均池化，输出一个 (C, 1, 1) 形状的张量，其中 C 是通道数。这个操作是将每个通道的空间信息汇聚成一个标量，代表该通道的全局信息
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)# 通过一个 1x1 卷积层（也可以看作是一个全连接层），将全局平均池化的结果映射回原始的通道数。卷积操作的作用是通过学习，生成通道的注意力权重
        self.act = nn.Sigmoid() # 使用 Sigmoid 激活函数将权重压缩到 [0, 1] 之间，确保每个通道都有一个在该范围内的权重值
    def forward(self, x: torch.Tensor) -> torch.Tensor: #前向传播： 输入的特征图 x 经过全局平均池化、1x1 卷积和 Sigmoid 激活后，得到每个通道的注意力权重
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x))) # 将原始输入 x 和通道注意力权重相乘，对每个通道的输出进行加权，从而实现通道注意力的调整


class SpatialAttention(nn.Module): # 空间注意力模块
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""

        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

        # 首先，对输入 x 在通道维度（即第 1 维）上进行平均，得到一个代表每个空间位置的通道间平均值。然后，对输入 x 在通道维度上进行最大值操作，得到每个空间位置的最大响应
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

#==============================================================================
#焊点注意力机制：颜色引导的注意力模块

class ChannelAttention2(nn.Module):
    """改进后的通道注意力模块，结合全局平均池化和局部最大池化"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        #self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.local_pool = nn.AdaptiveMaxPool2d(1)  # 局部最大池化
        self.fc = nn.Conv2d(channels , channels, 1, 1, 0, bias=True)  # 合并两种池化特征
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """结合全局平均池化和局部最大池化，生成通道注意力权重"""
        # global_feat = self.global_pool(x)  # 全局平均池化
        local_feat = self.local_pool(x)  # 局部最大池化
        # combined_feat = torch.cat([global_feat, local_feat], dim=1)  # 合并特征
        attention = self.act(self.fc(local_feat))  # 生成通道注意力
        return x * attention  # 加权输入


class SpatialAttention2(nn.Module):
    """改进后的空间注意力模块，结合局部卷积特征"""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = kernel_size // 2  # 根据卷积核大小确定填充
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)  # 空间注意力卷积层
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """结合局部区域的平均值和最大值进行空间注意力计算"""
       # avg_pool = torch.mean(x, dim=1, keepdim=True)  # 计算每个位置的通道平均值
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # 计算每个位置的通道最大值
        #combined_feat = torch.cat([avg_pool, max_pool], dim=1)  # 合并特征
        attention = self.act(self.conv(max_pool))  # 生成空间注意力
        return x * attention  # 加权输入
class SolderCbam(nn.Module):
    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)  # 通道注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)  # 空间注意力模块
        self.H_only = CoordAtt_HOnly(c1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用改进后的通道和空间注意力"""
        x = self.H_only(x)
        x = self.channel_attention(x)  # 通道注意力
        x = self.spatial_attention(x)  # 空间注意力
        return x
# 高度方向上的注意力
class HCAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(HCAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 对高度方向进行池化

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 通道注意力部分，参考你提供的实现
        self.pool = nn.AdaptiveAvgPool2d(1)  # 对每个通道做全局平均池化
        self.fc = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # 高度方向的池化
        x_h = self.pool_h(x)

        # 使用1x1卷积生成高度方向的注意力图
        y = self.conv1(x_h)
        y = self.bn1(y)
        y = self.act(y)

        # 生成高度方向的注意力
        a_h = self.conv_h(y).sigmoid()
        # 通道注意力部分：全局平均池化
        channel_attention = self.pool(x)  # 对输入的特征图进行全局平均池化
        channel_attention = self.fc(channel_attention)  # 1x1卷积生成通道权重
        channel_attention = self.sigmoid(channel_attention)  # Sigmoid激活函数

        # 扩展通道注意力到空间维度
        channel_attention = channel_attention.squeeze(-1).squeeze(-1)  # 从 (n, c, 1, 1) -> (n, c)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)  # 恢复为 (n, c, 1, 1)

        # 应用高度方向和通道注意力
        out = identity * a_h * channel_attention

        return out

class  HCSAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(HCSAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 对每个通道进行高度方向池化

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 通道注意力部分，参考你提供的实现
        self.pool = nn.AdaptiveAvgPool2d(1)  # 对每个通道做全局平均池化
        self.fc = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

        # 新增空间注意力部分
        self.spatial_conv = nn.Conv2d(inp, 1, kernel_size=3, stride=1, padding=1)  # 空间上的细小特征注意力
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # 高度方向的池化
        x_h = self.pool_h(x)

        # 使用1x1卷积生成高度方向的注意力图
        y = self.conv1(x_h)
        y = self.bn1(y)
        y = self.act(y)

        # 生成高度方向注意力
        a_h = self.conv_h(y).sigmoid()

        # 通道注意力部分：对每个通道进行全局平均池化
        channel_attention = self.pool(x)  # 对输入的特征图进行全局平均池化
        channel_attention = self.fc(channel_attention)  # 1x1卷积生成通道权重
        channel_attention = self.sigmoid(channel_attention)  # Sigmoid激活函数

        # 扩展通道注意力到空间维度
        channel_attention = channel_attention.squeeze(-1).squeeze(-1)  # 从 (n, c, 1, 1) -> (n, c)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)  # 恢复为 (n, c, 1, 1)

        # 空间方向上的细小焊点注意力图
        spatial_attention = self.spatial_conv(x)  # 使用3x3卷积生成空间注意力图
        spatial_attention = self.spatial_sigmoid(spatial_attention)  # 使用Sigmoid激活函数

        # 将高度方向注意力、通道注意力和空间注意力结合
        out = identity * a_h * channel_attention * spatial_attention


class CoordAtt_HOnly(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt_HOnly, self).__init__()
        oup = inp
        # 只处理高度方向，不需要宽度方向的池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 对高度方向进行池化

        mip = max(8, inp // reduction)

        # 使用1x1卷积处理池化后的特征图
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 只需要生成高度方向的注意力图
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # 对输入进行高度方向的池化
        x_h = self.pool_h(x)

        # 使用1x1卷积生成高度方向的注意力图
        y = self.conv1(x_h)
        y = self.bn1(y)
        y = self.act(y)

        # 生成高度方向的注意力
        a_h = self.conv_h(y).sigmoid()

        # 将生成的高度注意力应用到输入特征图上
        out = identity * a_h

        return out
#通道叠加 最大池化
class HCAtt_Max(nn.Module):
    def __init__(self, inp, reduction=32):
        super(HCAtt_Max, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 对高度方向进行池化

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 通道注意力部分
        self.pool = nn.AdaptiveAvgPool2d(1)  # 对每个通道做全局平均池化
        self.fc = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

        # 添加最大池化操作
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 对每个通道做全局最大池化

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # 高度方向的池化
        x_h = self.pool_h(x)
        # 使用1x1卷积生成高度方向的注意力图
        y = self.conv1(x_h)
        y = self.bn1(y)
        y = self.act(y)
        # 生成高度方向的注意力
        a_h = self.conv_h(y).sigmoid()

        # 通道注意力部分：全局平均池化和最大池化
        avg_pool_attention = self.pool(x)  # 对输入的特征图进行全局平均池化
        max_pool_attention = self.max_pool(x)  # 对输入的特征图进行全局最大池化

        # 将两者融合
        combined_attention = torch.cat([avg_pool_attention, max_pool_attention], dim=1)

        # 1x1卷积生成最终的通道注意力
        channel_attention = self.fc(combined_attention)
        channel_attention = self.sigmoid(channel_attention)  # Sigmoid激活函数

        # 扩展通道注意力到空间维度
        channel_attention = channel_attention.squeeze(-1).squeeze(-1)  # 从 (n, c, 1, 1) -> (n, c)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)  # 恢复为 (n, c, 1, 1)

        # 应用高度方向和通道注意力
        out = identity * a_h * channel_attention

        return out
#高度方向叠加最大池化
class H_MAx_C_MaxAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(H_MAx_C_MaxAtt, self).__init__()
        oup = inp
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))  # 对高度方向进行平均池化
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))  # 对高度方向进行最大池化

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 通道注意力部分，参考你提供的实现
        self.pool = nn.AdaptiveAvgPool2d(1)  # 对每个通道做全局平均池化
        self.fc = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)  # 1x1卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        # 添加最大池化操作
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 对每个通道做全局最大池化

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        # 高度方向的池化
        x_h_avg = self.pool_h_avg(x)  # 平均池化
        x_h_max = self.pool_h_max(x)  # 最大池化

        # 合并平均池化和最大池化的结果
        x_h = x_h_avg + x_h_max  # 或者使用拼接：torch.cat([x_h_avg, x_h_max], dim=1)

        # 使用1x1卷积生成高度方向的注意力图
        y = self.conv1(x_h)
        y = self.bn1(y)
        y = self.act(y)

        # 生成高度方向的注意力
        a_h = self.conv_h(y).sigmoid()

        # 通道注意力部分：全局平均池化和最大池化
        avg_pool_attention = self.pool(x)  # 对输入的特征图进行全局平均池化
        max_pool_attention = self.max_pool(x)  # 对输入的特征图进行全局最大池化

        # 将两者融合（可以加权相加，也可以拼接）
        combined_attention = avg_pool_attention + max_pool_attention  # 或者你可以使用拼接方式：torch.cat([avg_pool_attention, max_pool_attention], dim=1)

        # 1x1卷积生成最终的通道注意力
        channel_attention = self.fc(combined_attention)
        channel_attention = self.sigmoid(channel_attention)  # Sigmoid激活函数

        # 扩展通道注意力到空间维度
        channel_attention = channel_attention.squeeze(-1).squeeze(-1)  # 从 (n, c, 1, 1) -> (n, c)
        channel_attention = channel_attention.unsqueeze(2).unsqueeze(3)  # 恢复为 (n, c, 1, 1)

        # 应用高度方向和通道注意力
        out = identity * a_h * channel_attention

        return out