import copy
import time
import torch
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model import VGG16
import torch.nn as nn


def train_val_data_process(data_dir='Rice_Image_Dataset', batch_size=64):
    # 定义数据预处理：VGG16 要求输入 224x224，并使用 ImageNet 的均值和标准差
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一缩放到 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet均值和标准差
    ])

    # 加载全部训练数据（从 train 文件夹）
    full_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)

    # 计算训练集和验证集的大小（这里按80%训练,20%验证）
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练集打乱
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def get_pretrained_vgg16(num_classes=5):
    # 加载预训练权重
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 获取分类器最后一层的输入特征数（原为 4096）
    in_features = model.classifier[6].in_features
    # 替换最后一层为五分类
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 当前时间
    since = time.time()
    writer = SummaryWriter('runs/vgg16_rice')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for data in train_dataloader:
            img, targets = data  # targets 只包含 0 和 1 两个值。

            img = img.to(device)  # 这是因为 ImageFolder 会根据子文件夹名称的字母顺序自动分配类别索引
            targets = targets.to(device)
            model.train()

            output = model(img)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)#dim 参数指定了沿着哪个维度计算最大值
            # 计算每一个batch的损失函数
            loss = criterion(output, targets)

            # 利用梯度更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 对损失函数进行累加
            train_loss += loss.item() * img.size(0)
            # img 是一个四维张量，形状通常为 (batch_size, channels, height, width)。
            # 在 PyTorch 中，size(0) 返回张量第一个维度的大小，也就是 batch_size。
            # 将平均损失 loss.item() 乘以批次大小，就得到了该批次的总损失（即该批次所有样本的损失之和）。
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == targets.data)
            # 当前用于训练的样本数量
            train_num += img.size(0)

            # 计算并保存验证集的loss值
        with torch.no_grad():
            for data in val_dataloader:
                img, targets = data
                img = img.to(device)
                targets = targets.to(device)
                model.eval()

                output = model(img)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, targets)

                val_loss += loss.item() * img.size(0)
                val_corrects += torch.sum(pre_lab == targets.data)
                val_num += img.size(0)

        # 计算平均损失和准确率
        epoch_train_loss = train_loss / train_num
        epoch_train_acc = train_corrects.double().item() / train_num
        epoch_val_loss = val_loss / val_num
        epoch_val_acc = val_corrects.double().item() / val_num

        print(f"{epoch} train loss:{epoch_train_loss:.4f} train acc: {epoch_train_acc:.4f}")
        print(f"{epoch} val loss:{epoch_val_loss:.4f} val acc: {epoch_val_acc:.4f}")

        # 使用TensorBoard绘制训练和验证的曲线
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        if epoch_val_acc > best_acc:
            # 保存当前最高准确度
            best_acc = epoch_val_acc
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print(f"训练和验证耗费的时间{time_use // 60:.0f}m{time_use % 60:.0f}s")

    writer.close()

    # 加载最佳模型并保存
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "best_model_rice_pretrained.pth")


if __name__ == '__main__':
    # 加载需要的模型
    model1 = get_pretrained_vgg16(num_classes=5)
    # 加载数据集
    train_data, val_data = train_val_data_process()

    # 利用现有的模型进行模型的训练
    train_model_process(model1, train_data, val_data, num_epochs=10)
