import torch
from torch import nn
from torchsummary import summary


class ChannelAttention(nn.Module):  # 通道注意力模块
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 无论输入特征图的高度和宽度是多少，输出都将是 (batch, channels, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 全连接层Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            # 这里将通道数先降为原来的 1/reduction，以减少参数量和计算量，然后再升回原通道数。这是一种“瓶颈”设计。
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 先全局平均池化，再把结果放入全连接层
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)  # Sigmoid 归一化到 [0,1]
        return out


# 平均池化捕获了全局的、平滑的响应；最大池化则保留了最突出的局部特征。两者结合可以为后续的卷积层提供更丰富的空间信息。
class SpatialAttention(nn.Module):  # 空间注意力模块
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        # padding=kernel_size//2：由于步长默认是 1，这样的填充可以保证卷积操作不改变特征图的高度和宽度，即输出尺寸与输入相同
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上计算平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # dim=1：在通道维度上计算所有通道的平均值，因为特征图的形状为 (batch_size, channels, height, width)
        # keepdim=True：保持输出张量的维度数与输入相同，避免因降维而丢失维度。
        # 若不设置，输出形状会变成 (batch, height, width)，而保持维度后形状为 (batch, 1, height, width)。

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 对输入特征图在通道维度上分别计算平均池化和最大池化，得到两个单通道特征图，然后将它们拼接在一起
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 拼接后为 2 通道
        out = self.sigmoid(self.conv(x_cat))
        return out


class CBAM(nn.Module):  # CBAM模块将通道注意力 + 空间注意力串联
    def __init__(self, in_channels, reduction=16, kernel_size=3):  # kernel_size：空间注意力中卷积核的大小（默认为3），用于生成空间权重图
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力，再应用空间注意力
        out = self.channel_attention(x) * x  # self.channel_attention(x)的形状为 (batch, in_channels, 1, 1)，x 是输入特征图
        out = self.spatial_attention(out) * out  # self.spatial_attention(out) 的输出形状为 (batch, 1, H, W)
        return out


# 为什么先应用通道注意力，再应用空间注意力？
# 通道注意力关注“什么”（what）是重要的特征（如颜色、纹理对应的通道），
# 而空间注意力关注“哪里”（where）是重要的位置。先进行通道重标定，可以让后续的空间注意力在更有判别力的特征图上计算权重，从而提升效果


class Residual(nn.Module):  # Residual残差块 定义
    def __init__(self, input_channels, output_channels, use_1conv=False, strides=1, use_cbam=True):
        """
            input_channels: 输入通道数
            output_channels: 输出通道数（第二个卷积的输出通道）
            use_1conv: 是否使用 1x1 卷积调整 shortcut 以匹配维度
            strides: 第一个卷积的步长（用于下采样）
            use_cbam: 是否在该残差块后添加 CBAM 模块
        """
        super(Residual, self).__init__()
        # 第一个卷积：可能改变通道数和尺寸
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1,
                               stride=strides)

        self.bn1 = nn.BatchNorm2d(output_channels)  # output_channels 是该BN层期望的输入通道数。Batch Normalization不会改变特征图的通道数和空间尺寸
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # 如果需要对 shortcut 进行变换，shortcut 是指那条“跳过”卷积层的直接连接路径，则定义 1x1 卷积
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1,
                                   stride=strides)
        else:
            self.conv3 = None

        # 可选 CBAM 模块
        self.use_cbam = use_cbam
        # 将传入的参数 use_cbam 保存为实例属性，为了在 forward 中也能访问到这个配置信息
        if use_cbam:
            self.cbam = CBAM(output_channels)

    def forward(self, x):
        # 使用BasicBlock结构，两个3×3卷积。而Bottleneck 用于更深的 ResNet（如 ResNet-50/101/152），其结构为 1×1 → 3×3 → 1×1
        identity = x  # 保存 shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要，对 shortcut 进行变换
        if self.conv3 is not None:
            identity = self.conv3(x)  # 注意使用原始输入 x

        # 判断是否启用 CBAM 注意力机制
        if self.use_cbam:
            out = self.cbam(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet32(nn.Module):
    """
    ResNet32 架构：
    初始卷积：3x3, 16 通道，步长 1，无池化
    阶段1：5 个残差块，16 通道，所有块步长为 1（无下采样）
    阶段2：5 个残差块，32 通道，第一个块步长为 2（下采样），其余步长 1
    阶段3：5 个残差块，64 通道，第一个块步长为 2（下采样），其余步长 1
    全局平均池化 + 全连接层
    每个残差块均包含 CBAM 注意力
    """

    def __init__(self, num_classes):
        super(ResNet32, self).__init__()

        # 初始卷积层（适合 32x32 输入）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 阶段1：5 个残差块，16 通道
        self.layer1 = self._make_layer(16, 16, blocks=5, stride=1, use_cbam=True)

        # 阶段2：5 个残差块，32 通道，第一个块下采样
        self.layer2 = self._make_layer(16, 32, blocks=5, stride=2, use_cbam=True)

        # 阶段3：5 个残差块，64 通道，第一个块下采样
        self.layer3 = self._make_layer(32, 64, blocks=5, stride=2, use_cbam=True)

        # 全局平均池化和分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#自适应平均池化，只需要指定输出尺寸，由 PyTorch 自动计算池化方式，输出尺寸固定。
        self.fc = nn.Linear(64, num_classes)

    # 在 Python 类中，方法的定义顺序不影响它们在实例化时的可用性。类定义会先完整地解析所有方法（包括 _make_layer），然后才执行 __init__ 方法。
    def _make_layer(self, in_channels, out_channels, blocks, stride, use_cbam):
        """
        构建一个阶段，包含多个残差块
        Args:
            in_channels: 该阶段输入通道数
            out_channels: 该阶段输出通道数（即残差块的输出通道）
            blocks: 残差块数量
            stride: 第一个残差块的步长（用于下采样）
            use_cbam: 是否在该阶段的所有残差块中使用 CBAM
        """
        layers = []  # 临时存储当前阶段（stage）中所有残差块（Residual 实例），以便最终将它们组合成一个 nn.Sequential 容器。

        # 第一个残差块可能需要下采样和 shortcut 变换
        # shortcut（也称为“跳连接”或“恒等映射”）是残差块中的一种连接方式，它将块的输入直接加到块的输出上
        layers.append(Residual(in_channels, out_channels,
                               use_1conv=(stride != 1 or in_channels != out_channels),
                               # 只要上述任一条件成立，就设置 use_1conv=True，表示需要额外的 1×1 卷积来调整 shortcut
                               strides=stride,
                               use_cbam=use_cbam))
        # 后续残差块：输入通道已变为 out_channels，步长为 1，无需 shortcut 变换
        for _ in range(1, blocks):
            layers.append(Residual(out_channels, out_channels,
                                   use_1conv=False,
                                   strides=1,
                                   use_cbam=use_cbam))
        return nn.Sequential(*layers)# _ 可以明确告诉阅读代码的人：“这里我们只需要循环次数，不关心循环索引的值”

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积：3→16，32×32
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 阶段1：16→16，32×32
        x = self.layer2(x)  # 阶段2：16→32，16×16
        x = self.layer3(x)  # 阶段3：32→64，8×8

        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)
        x = self.fc(x)  # 全连接分类：64 → num_classes
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet32(num_classes=15).to(device)
    print(summary(model, (3, 32, 32)))
    # 注意力机制只影响模型内部结构，数据加载部分完全独立
