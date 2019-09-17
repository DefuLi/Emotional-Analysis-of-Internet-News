# 互联网新闻情感分析

## 1 赛题简介
赛题简介：[“互联网新闻情感分析”](https://www.datafountain.cn/competitions/350)赛题，是CCF大数据与计算智能大赛赛题之一。对新闻情绪进行分类，0代表正面情绪、1代表中性情绪、2代表负面情绪。<br>
程序简介：开发工具是pycharm，使用GPU加速。所使用的关键库有pytorch、torchtext、numpy、pandas、visdom等。<br>

## 2 项目结构
项目文件夹共包括以下文件及文件夹：<br>
main.py 主程序，运行此程序模型开始训练以及测试。<br>
net.py 定义网络结构，采用LSTM神经网络，最后一层是全连接层。<br>
preprocess.py 预处理程序，对官方提供的csv文件进行处理，包括清理、分词、拆分数据集等工作。<br>
trainfiles 存储预处理过程中与训练集有关的处理文件。<br>
testfiles 存储预处理过程中与测试集有关的处理文件。<br>
torchtextfiles 存储拆分好的训练集、验证集和测试集，供torchtext加载。<br>
wordfiles 存储停用词文件和词向量文件。<br>

## 3 预处理
官方提供的数据集有Train_DataSet.csv、Train_DataSet_Label.csv、Test_DataSet.csv。由于Train_DataSet.csv和Train_DataSet_Label.csv数据集中id字段不一致、三个数据集存在较多标点符号和无用符号、存在停用词、存在title和content字段分开等问题，所以在预处理阶段所做的主要工作有：提取共有的内容、清理数据集的标点符号和英文字符、对数据集进行分词、合并title和content字段、拆分数据集为训练集验证集和测试集等。<br>
最后处理好的数据集包括train.csv、val.csv和test.csv，存放在torchtextfiles文件夹中。<br>
```python
    针对官方训练集的处理示例
    train_id = train_id('trainfiles/Train_DataSet.csv', 'trainfiles/Train_DataSet_Label.csv')
    train_dataset = train_dataset('trainfiles/Train_DataSet.csv', 'trainfiles/Train_DataSet_Label.csv', train_id)
    to_train = pd.DataFrame(train_dataset)
    to_train.to_csv('trainfiles/Train.csv', index=False, header=['id', 'title', 'content', 'label'])

    train_clear = train_clear('trainfiles/Train.csv')
    to_cleartext = pd.DataFrame(train_clear)
    to_cleartext.to_csv('trainfiles/Train_Clear.csv', index=False, header=['id', 'title', 'content', 'label'])

    train_word = sent2word('trainfiles/Train_Clear.csv')
    to_word = pd.DataFrame(train_word)
    to_word.to_csv('trainfiles/Train_Word.csv', index=False, header=['id', 'title', 'content', 'label'])

    train_joint = joint_titlecontent('trainfiles/Train_Word.csv')
    to_joint = pd.DataFrame(train_joint)
    to_joint.to_csv('trainfiles/Train_Joint.csv', index=False, header=['id', 'text', 'label'])
```

## 4 网络结构
采用的LSTM神经网络进行分类，网络层次依次是嵌入层，LSTM层、全连接层。嵌入层使用300维的词嵌入向量表示，h/c长度是128，全连接层输出是3。其余超参数设置查看main.py文件。<br>
```python
定义神经网络结构
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Net, self).__init__()

        # 单词=>[300]
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # [embedding_dim]=>[hidden_dim]
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        # [hidden_dim*2]=>[3]
        self.fc = nn.Linear(hidden_dim * 2, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x:[seq_len,b,1]=>[seq_len,b,embedding_dim]
        embedding = self.dropout(self.embedding(x))

        # out: [seq_len, b, hid_dim*2]
        # h: [num_layers*2, b, hid_dim]=>[4,b,hidden_dim]
        # c: [num_layers*2, b, hid_dim]
        out, (h, c) = self.lstm(embedding)
        h = torch.cat([h[-2], h[-1]], dim=1)

        h = self.dropout(h)
        output = self.fc(h)

        return output
```

## 5 主程序训练验证和测试
共进行了50个epoch训练，在最后一个epoch上的分类准确率为0.97，验证集上的分类准确率为0.75，提交到官网的测试结果为0.69，该结果是用F1值计算得出。<br>
我认为该模型过拟合了。如需要请进行调参。<br>

## 6 注意事项
本程序采用GPU加速，如果不使用GPU加速，请在main.py文件中删除相关语句。<br>
本程序使用了visdom可视化工具，如果你没有安装该工具，可以在终端安装，并开启服务使用。如不使用，也可以在main.py文件中删除相关语句。<br>
本程序使用了torchtext库，方便建立词典，shuffle等操作。<br>
由于github对上传文件大小的限制，位于wordfiles文件夹中的词向量文件没有上传，如需要在该[链接](https://pan.baidu.com/s/18T6DRVmS_cZu5u64EbbESQ)中下载，并放在wordfiles文件中。<br>
