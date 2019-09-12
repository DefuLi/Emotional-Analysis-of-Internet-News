import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
import pandas as pd
import numpy as np
import visdom
import torchtext
from torchtext import data

FIX_CHARS = 200  # 句子的最大长度
EPOCHS = 50  # epoch的个数
BATCH_SIZE = 64  # batch大小
HIDDEN_DIM = 128  # 隐藏层长度
LR = 1e-4  # 优化器学习率
WEIGHT_DECAY = 1e-2  # 优化器衰减系数
BEST_ACC = 0  # 最优的准确率
BEST_EPOCH = 0  # 最优的epoch

device = torch.device('cuda')  # GPU加速


def tokenizer_text(text):
    """
    Field的tokenize，见split2batches函数。
    :param text:
    :return:
    """
    return text.split()


def tokenizer_id(text):
    """
    不处理id文本。
    :param text:
    :return:
    """
    return text


def tokenizer_label(id):
    """
    返回int型的id。
    :param id:
    :return:
    """
    return int(id)


def getVocab():
    """
    返回TEXT，里面有词向量的权重等参数。
    :return:
    """

    IDX = data.Field(sequential=False, tokenize=tokenizer_id, use_vocab=True)  # sequential=False就不用拆分该字段内容，保留整体。
    TEXT = data.Field(sequential=True, tokenize=tokenizer_text, use_vocab=True, fix_length=FIX_CHARS)
    LABEL = data.Field(sequential=False, tokenize=tokenizer_label, use_vocab=False, dtype=torch.long)

    train, val, test = data.TabularDataset.splits(path='torchtextfiles', train='train.csv', validation='val.csv',
                                                  test='test.csv', format='csv', skip_header=True,
                                                  fields=[('id', IDX), ('text', TEXT), ('label', LABEL)])

    vects = torchtext.vocab.Vectors(name='wordfiles/Sgns.context.word-word.dynwin5.thr10.neg5.dim300.iter5.txt',
                                    cache='cachefiles/')
    TEXT.build_vocab(train, vectors=vects)

    IDX.build_vocab(test)

    """词向量测试用例
    print(type(TEXT))
    print(type(TEXT.vocab))
    print(TEXT.vocab.itos[:200])  # 显示前200个词语
    print(TEXT.vocab.vectors[98583])  # 显示'德福'的词向量
    """

    return IDX, TEXT, train, val, test


def split2batches(batch_size=BATCH_SIZE, device='cpu'):
    """
    按批次分割，生成迭代器。
    :param batch_size:
    :param device:
    :return:
    """

    _, _, train, val, test = getVocab()

    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), batch_size=batch_size,
                                                                 device=device, sort_key=lambda x: len(x.text),
                                                                 sort_within_batch=False)

    """输出第一个examples的text和label
    print(train_iter)
    print(train_iter.dataset.examples[1].text)
    print(train_iter.dataset.examples[1].label)
    """

    return train_iter, val_iter, test_iter


def accrate(pre, y):
    """
    求分类准确率。
    :param pre:
    :param y:
    :return:
    """
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, y).sum().float().item()
    acc = correct / float(y.size(0))
    return acc


def evalute(net, val_iter):
    """
    计算验证集的准确率。
    :param net:
    :param val_iter:
    :return:
    """
    net.eval()

    avg_acc = []
    for step, item in enumerate(val_iter):
        item.text = item.text.to(device)  # GPU加速
        item.label = item.label.to(device)

        with torch.no_grad():
            pre = net(item.text)
            acc = accrate(pre, item.label)
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()

    return avg_acc


def test(net, test_iter, IDX):
    """
    对官方的测试集进行测试。
    :param net:
    :param test_iter:
    :param IDX:
    :return:
    """
    temp = []
    id = []
    label = []
    for step, item in enumerate(test_iter):
        id.extend(item.id)
        item.text = item.text.to(device)  # GPU加速
        with torch.no_grad():
            pre = net(item.text)
            pre = pre.argmax(dim=1)  # 示例tensor([0,1,2,1,1,...,])
            pre = pre.cpu().data.numpy()
            pre = pre.tolist()  # 示例[0,1,2,1,1,...,]
            label.extend(pre)

    id = np.array(id)  # array([1,2,5,3,...])
    id = id.tolist()  # [1,2,5,3,...]

    for item in id:  # 将id中的数值转换为真实的id文本
        temp.append(IDX.vocab.itos[item])

    dataframe = pd.DataFrame({'id': temp, 'label': label})
    dataframe.to_csv('endclass.csv', index=False)

    return label


if __name__ == '__main__':

    IDX, TEXT, _, _, _ = getVocab()
    train_iter, val_iter, test_iter = split2batches()

    net = Net(len(TEXT.vocab), 300, HIDDEN_DIM).to(device)  # embedding_dim=300, hidden_dim=128
    pretrained_embedding = TEXT.vocab.vectors
    # print('这是参数的尺寸：',pretrained_embedding.shape)  # torch.Size([135296, 300])

    net.embedding.weight.data.copy_(pretrained_embedding)

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    vis = visdom.Visdom()
    vis.line([0], [-1], win='loss', opts=dict(title='loss'))
    vis.line([0], [-1], win='acc', opts=dict(title='acc'))
    flagstep = 0

    for epoch in range(EPOCHS):  # 共有EPOCHS个epoch，每个epoch包含len(epoch)/batch_size个batch，每个batch共有BATCH_SIZE条数据

        avg_acc = []
        for step, item in enumerate(train_iter):  # item.text.shape=>[100,8]  item.label.shape=>[8]

            item.text = item.text.to(device)  # GPU加速
            item.label = item.label.to(device)

            net.train()
            # [seq_len,b]=>[b,3]
            pre = net(item.text)
            # print(pre.shape, item.label.shape)
            # print(pre)
            # print(item.label)
            loss = criterion(pre, item.label)

            acc = accrate(pre, item.label)
            avg_acc.append(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                print('epoch:{},batch:{},acc:{},loss:{}'.format(epoch, step, acc, loss))
                vis.line([loss.item()], [flagstep], win='loss', update='append')  # visdom显示loss和acc曲线
                vis.line([acc], [flagstep], win='acc', update='append')
                flagstep += 1

        print('第{}个epoch的平均准确率为：{}'.format(epoch, np.array(avg_acc).mean()))

        if epoch % 1 == 0:
            val_acc = evalute(net, val_iter)
            if val_acc > BEST_ACC:
                BEST_EPOCH = epoch
                BEST_ACC = val_acc

                torch.save(net.state_dict(), 'best_par.pt')

    print('验证集最优的准确率为：{}'.format(BEST_ACC))
    print('验证集最优的epoch为：{}'.format(BEST_EPOCH))

    net.load_state_dict(torch.load('best_par.pt'))
    print('加载最优模型参数完成...')

    test(net, test_iter, IDX)
    print('分类完成，请查看输出文件！')
