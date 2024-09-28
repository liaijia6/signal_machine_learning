# 导入相关库
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # 切换到当前文件路径
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # 加载两个函数：load_img（从本地文件加载图片到代码里），img_to_array
import numpy as np # 矩阵运算
# from sklearn.model_selection import train_test_split # 划分训练集和测试集
from tensorflow.keras.models import Sequential # 神经网络模型
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # 神经网络层
from tensorflow.keras.optimizers import Adam # 优化函数
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # 评估指标
from sklearn.metrics import roc_curve # ROC曲线
import matplotlib.pyplot as plt # 画图库
import tensorflow as tf # 神经网络库


# 数据处理(图片转为向量)
# 本地数据文件路径：
# 训练集-YES：./data/train/YES/1.png, 2.png, .... 
# 训练集-NO：./data/train/NO/1.png, 2.png, .... 
# 测试集-YES：./data/test/YES/1.png, 2.png, ....
# 测试集-NO：./data/test/NO/1.png, 2.png, ....
# 训练代码所在路径：code/train.py

# 本地数据文件路径
train_yes_dir = './data/train/YES'
train_no_dir = './data/train/NO'
test_yes_dir = './data/test/YES'
test_no_dir = './data/test/NO'
image_size = (64, 64)  # 根据实际情况调整图像大小

# 加载数据
def load_data(yes_dir, no_dir, image_size):
    images = [] # 存储图像数据，是个空列表，例如[1,2,3,4,5,6,7,8,9,10]
    labels = [] # 存储图像标签
    
    def load_images_from_dir(directory, label):
        for image_name in os.listdir(directory):
            image_path = os.path.join(directory, image_name)
            try:
                image = load_img(image_path, target_size=image_size, color_mode='grayscale')
                image = img_to_array(image)
                images.append(image)
                labels.append(label)
            except Exception as e:
                continue
                # print(f"无法加载图像 {image_path}: {e}")
    
    # 加载 YES 类图像
    load_images_from_dir(yes_dir, 1)
    
    # 加载 NO 类图像
    load_images_from_dir(no_dir, 0)
    
    return np.array(images), np.array(labels)

# 加载训练集数据
X_train, y_train = load_data(train_yes_dir, train_no_dir, image_size)

# 加载测试集数据
X_test, y_test = load_data(test_yes_dir, test_no_dir, image_size)

# 归一化图像数据
# X_train.shape (1900, 64, 64, 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
# print("X_train:", X_train)
# [1,2,3,4,5,6,7,8,9,10]  维度: 1 * 10
# [ [1,2],[3,4] ] 维度: 2 * 2
# [ [ [1,2],[3,4] ], [ [5,6],[7,8] ] ] 维度: 2 * 2 * 2

# X_train.shape (1900, 64, 64, 1)
# 解释:
# 1900: 样本数量，即训练集中有1900张图像
# 64: 图像的高度，像素为64
# 64: 图像的宽度，像素为64
# 1: 图像的通道数，这里是灰度图像，所以通道数为1

# 1： integer;  1.0000000000000001: float




################ 创建模型（神经网络）################
# 构建模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


################ 模型训练 ################
model.fit(X_train, y_train, epochs=12, batch_size=64, validation_split=0.1)
# 如果有2万个样本，batch_size=32，那么每个epoch会处理20000/32个batch，每个batch处理32个样本。
# 如果有2千个样本，batch_size=32，那么每个epoch会处理2000/32个batch，每个batch处理32个样本。
# batch size一般都是 2的指数次方。
# 第一组实验：batch_size=32，epoch=10, 训练集准确率90.99%
# 第2组实验：batch_size=64，epoch=10, 训练集准确率92.99%
# 第3组实验：batch_size=128，epoch=10, 训练集准确率92.99%

################ 模型评估（验证模型的准确率） ################
y_pred = (model.predict(X_test) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict(X_test).ravel())

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {auc:.2f}")

################ 绘制曲线（可视化测试结果，例如AUC） ################
y_pred_prob = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()