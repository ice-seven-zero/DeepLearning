import copy
import time
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from model import ResNet32


def load_data(data_dir='Vegetable Images', batch_size=64, img_size=224):#应该是32，
    # 但是加了self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),  # 随机裁剪并缩放
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转 ±10 度
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  #使用 ImageNet 的均值和标准差
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集预处理（不增强）
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 直接缩放到固定尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（ImageFolder 会自动根据子文件夹名分配类别索引）
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'validation'), transform=val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)#启用 4 个子进程来并行加载数据
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 优化器与损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    # 记录最佳模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    writer = SummaryWriter('runs/resnet32_vegetable_CBAM')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}/{num_epochs - 1}")
        print("-" * 30)

        # 训练阶段
        model.train()
        train_loss = 0
        train_corrects = 0
        train_num = 0

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_corrects += torch.sum(preds == targets)
            train_num += images.size(0)

        epoch_train_loss = train_loss / train_num
        epoch_train_acc = train_corrects.double().item() / train_num

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_num = 0

        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_corrects += torch.sum(preds == targets)
                val_num += images.size(0)

        epoch_val_loss = val_loss / val_num
        epoch_val_acc = val_corrects.double().item() / val_num

        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("保存最佳模型权重")

        # 打印耗时
        time_elapsed = time.time() - since
        print(f"累计耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    writer.close()
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")

    # 加载最佳模型并保存
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "best_model_resnet32_vegetable_CBAM.pth")
    print("最佳模型已保存为 best_model_resnet32_vegetable_CBAM.pth")

    return model


if __name__ == '__main__':
    # 参数设置
    train_loader, val_loader = load_data(data_dir='Vegetable Images', batch_size=64)

    model = ResNet32(num_classes=15)
    trained_model = train_model_process(model, train_loader, val_loader, num_epochs=50)

