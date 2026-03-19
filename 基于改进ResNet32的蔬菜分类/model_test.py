import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from model import ResNet32  # 导入模型定义

def test_data_process(data_dir='Vegetable Images', batch_size=64, img_size=224):

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'),transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=4)
    return test_loader

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += torch.sum(preds == targets).item()
            total += targets.size(0)

    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f}")
    return accuracy

if __name__ == '__main__':
    test_loader = test_data_process(batch_size=64)
    model = ResNet32(num_classes=15)

    model.load_state_dict(torch.load('best_model_resnet32_vegetable_CBAM.pth', map_location='cpu'))

    test_model(model, test_loader)
    #测试集准确率: 0.9937