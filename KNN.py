import numpy as np
import cv2
import os
import time
import psutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 加载图像数据并进行预处理
def load_and_preprocess_images(folder):
    images = []
    labels = []
    for label in ['6', '9']:
        folder_path = os.path.join(folder, label)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (28, 28))  # 调整图像大小为28x28
                img = img.flatten()  # 展平图像
                images.append(img)
                labels.append(int(label))
    return np.array(images), np.array(labels)


# 指定图像数据路径
train_images_path = 'D:/runa/train_69/train'
test_images_path = 'D:/runa/train_69/test'

# 加载训练数据
X_train, y_train = load_and_preprocess_images(train_images_path)
# 加载测试数据
X_test, y_test = load_and_preprocess_images(test_images_path)

# 打印数据形状以确认预处理结果
print('Train Images shape:', X_train.shape)
print('Train Labels shape:', y_train.shape)
print('Test Images shape:', X_test.shape)
print('Test Labels shape:', y_test.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')

start_time = time.time()
for i in range(1000000):
    pass
end_time = time.time()
execution_time = end_time - start_time
print(f'code run time:{execution_time} s')


# 获取当前进程的内存信息
process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 10**6} MB")

print(f"CPU usage:, {process.cpu_percent(interval=1)} %")

def predict_new_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28)).flatten().reshape(1, -1)  # 将图像展平并调整形状
    prediction = knn.predict(img)
    return prediction[0]


# test path
new_image_path = 'D:/runa/MachineMeter_20240612_180_0_0_6.jpg'
print(f'The new image is predicted to be: {predict_new_image(new_image_path)}')
