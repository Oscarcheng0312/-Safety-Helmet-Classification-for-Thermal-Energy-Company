import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label in ['6', '9']:
            folder = os.path.join(folder_path, label)
            for filename in os.listdir(folder):
                if filename.endswith('.jpg'):
                    self.image_paths.append(os.path.join(folder, filename))
                    self.labels.append(0 if label == '6' else 1)
        print(f"Loaded {len(self.image_paths)} images from {folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def visualize_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()  # 转换为(H, W, C)格式
        image = (image * 0.5 + 0.5) * 255  # 反归一化
        axes[i].imshow(image.astype(np.uint8))
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.show()

# 图像预处理和数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomDataset('D:/runa/train_69/train', transform=train_transform)
test_dataset = CustomDataset('D:/runa/train_69/test', transform=test_transform)

# 可视化部分训练数据
visualize_sample_images(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class FineTunedDenseNet(nn.Module):
    def __init__(self):
        super(FineTunedDenseNet, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 2)  # 修改最后一层分类层

    def forward(self, x):
        x = self.densenet(x)
        return x

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    net = FineTunedDenseNet()
    net.to(device)  # 将模型移动到GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.0001)  # 使用AdamW优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    epoch_losses = []
    train_accuracies = []
    test_accuracies = []
    num_epochs = 30

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        print(f'Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.2f}%')

        scheduler.step(epoch_loss)  # 更新学习率

    print('Finished Training')

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='x', label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.savefig('D:/runa/train_69/densenet_training.png')
    plt.show()

    accuracy = test_accuracies[-1]
    print(f'Final Accuracy of the network on the test images: {accuracy:.2f}%')

    model_save_path = 'D:/runa/train_69/densenet_model.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'accuracy': accuracy
    }, model_save_path)
    print(f'Model and accuracy saved to {model_save_path}')
