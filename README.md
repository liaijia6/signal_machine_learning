# 基于机器学习的峰形分类

一、安装
```sh
conda activate base # 进入base环境
pip install keras # 安装keras
pip install --ignore-installed --upgrade tensorflow # 安装TensorFlow的纯CPU版本
pip install numpy scikit-learn matplotlib # 安装几个常用计算和绘图库
```

二、数据下载
进入链接下载：https://figshare.com/articles/dataset/Data-Envents/19246224?file=34200345
数据文件组织目录:
- 训练集: ./data/train/YES 和 ./data/train/NO
- 测试集: ./data/test/YES 和 ./data/test/NO

三、开始训练

打开命令行软件(iTerm2)，然后执行如下命令即可开始训练：
```sh
cd ~/Documents/signal_machine_learning/code/ && python3 train.py
```

注：
- 本代码用于参考，实际实验过程中为了提升准确率，包含更多的参数调试过程，请读者自行修改代码中参数进行实验
- 以上命令部分来自文档：Conda环境下安装Keras同时安装Tensorflow: https://blog.csdn.net/qq_45860901/article/details/127830239
