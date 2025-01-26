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


# 图像处理函数
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


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        # 确定第一个全连接层的输入大小
        self.f6 = nn.Linear(128 * 2 * 2, 2048)  # 这里的128 * 2 * 2取决于前面的卷积层和池化层的输出尺寸
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(self.ReLU(self.c2(x)))
        x = self.s3(self.ReLU(self.c3(x)))
        x = self.ReLU(self.c4(x))
        x = self.s5(self.ReLU(self.c5(x)))
        x = self.flatten(x)
        print(x.shape)  # 打印张量形状
        x = self.f6(x)
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x

# 检查是否有GPU可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    net = MyAlexNet()
    net.to(device)  # 将模型移动到GPU

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    epoch_losses = []
    num_epochs = 100

    # 训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Average Loss: {epoch_loss:.4f}')

    print('Finished Training')

    # 绘制每个epoch的loss图像
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('D:/runa/train_69/model/training_loss.png')
    plt.show()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy} %')

    # 保存整个模型
    model_save_path = 'D:/runa/train_69/model/alexnet_model.pth'
    torch.save(net, model_save_path)
    print(f'Model and accuracy saved to {model_save_path}')
