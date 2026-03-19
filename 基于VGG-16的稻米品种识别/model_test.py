import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


def get_pretrained_vgg16(num_classes=5):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def test_data_process(data_dir='Rice_Image_Dataset', batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'),transform=transform )

    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=2)
    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        for img, targets in test_dataloader:
            img = img.to(device)
            targets = targets.to(device)

            outputs = model(img)
            preds = torch.argmax(outputs, dim=1)
            test_corrects += torch.sum(preds == targets)
            test_num += img.size(0)

    test_acc = test_corrects.double().item() / test_num
    print(f"测试集准确率: {test_acc:.4f}")


if __name__ == "__main__":
    model = get_pretrained_vgg16(num_classes=5)

    model.load_state_dict(torch.load('best_model_rice_pretrained.pth'))

    test_dataloader = test_data_process(batch_size=64)

    test_model_process(model, test_dataloader)
    #测试集准确率: 0.9996，因为微调之后的模型太大，所以没上传github