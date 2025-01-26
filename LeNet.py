import os
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn.functional as F

def resize_with_padding(image, target_size):
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image
    image = cv2.resize(image, (new_size[1], new_size[0]))

    # Create a new image and paste the resized on it
    delta_w = target_size - new_size[1]
    delta_h = target_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_image

def process_image_folder(input_folder, output_folder, target_size=64):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for label in ['6', '9']:
        input_label_folder = os.path.join(input_folder, label)
        output_label_folder = os.path.join(output_folder, label)

        if not os.path.exists(output_label_folder):
            os.makedirs(output_label_folder)

        for filename in os.listdir(input_label_folder):
            if filename.endswith('.jpg'):
                img_path = os.path.join(input_label_folder, filename)
                image = cv2.imread(img_path)
                resized_image = resize_with_padding(image, target_size)
                output_path = os.path.join(output_label_folder, filename)
                cv2.imwrite(output_path, resized_image)

# 处理训练和测试文件夹
process_image_folder('D:/runa/train_69/train', 'D:/runa/train_69/train_resized')
process_image_folder('D:/runa/train_69/test', 'D:/runa/train_69/test_resized')

# 定义数据集和数据加载器
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomDataset('D:/runa/train_69/train_resized', transform=train_transform)
test_dataset = CustomDataset('D:/runa/train_69/test_resized', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 定义LeNet5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # 修改输入通道数为3
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, 120)  # 修改为16个14x14特征图
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 最后一层输出2个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))  # C1: Conv Layer
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # S2: Pooling Layer
        x = F.relu(self.conv2(x))  # C3: Conv Layer
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # S4: Pooling Layer
        x = x.view(-1, 16 * 14 * 14)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # C5: Fully Connected Layer
        x = F.relu(self.fc2(x))  # F6: Fully Connected Layer
        x = self.fc3(x)  # Output Layer
        return x

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    net = LeNet5()
    net.to(device)  # 将模型移动到GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)  # 使用Adam优化器
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=30)

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

        scheduler.step()  # 更新学习率

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

    plt.savefig('D:/runa/train_69/model/LeNet_training1.png')
    plt.show()

    accuracy = test_accuracies[-1]
    print(f'Final Accuracy of the network on the test images: {accuracy:.2f}%')

    model_save_path = 'D:/runa/train_69/model/LeNet_model1.pth'
    # 保存模型参数和准确率
    torch.save({
        'model_state_dict': net.state_dict(),
        'accuracy': accuracy
    }, model_save_path, _use_new_zipfile_serialization=False)
    print(f'Model state dict and accuracy saved to {model_save_path}')
