import pandas as pd
import codecs
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
import re
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# reference from : https://zhuanlan.zhihu.com/p/388009679

# 标签
news_label = [int(x.split('_!_')[1])-100 for x in codecs.open('toutiao_cat_data.txt')]
# 文本
news_text = [x.strip().split('_!_')[-1] if x.strip()[-3:] != '_!_' else x.strip().split('_!_')[-2] for x in codecs.open('toutiao_cat_data.txt')]



# 划分为训练集和验证集
th1 = 30000 # select only 3w
# stratify 按照标签进行采样，训练集和验证部分同分布
x_train, x_test, train_label, test_label =  train_test_split(news_text[:th1], news_label[:th1], test_size=0.2, stratify=news_label[:th1])
print(len(x_train), len(x_test))

# 分词器，词典
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=64)
print('completed tokenization')


# 数据集读取
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encoding, train_label)
test_dataset = NewsDataset(test_encoding, test_label)
print('completed NewsDataset')



# 步骤4：定义Bert模型
# downoad from : https://www.modelscope.cn/models/tiansz/bert-base-chinese
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is {0}'.format(device))
model.to(device)

# 单个读取到批量读取
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 优化方法
optim = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1
scheduler = get_linear_schedule_with_warmup(optim,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
print('completed model creation')

# 步骤5：模型训练与验证
# 训练函数
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optim.step()
        scheduler.step()

        iter_num += 1
        if (iter_num % 100 == 0):
            print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
            epoch, iter_num, loss.item(), iter_num / total_iter * 100))

    print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in test_dataloader:
        with torch.no_grad():
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs[0]
        logits = outputs[1]

        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("Accuracy: %.4f" % (avg_val_accuracy))
    print("Average testing loss: %.4f" % (total_eval_loss / len(test_dataloader)))
    print("-------------------------------")


for epoch in range(4):
    print("------------Epoch: %d ----------------" % epoch)
    train()
    validation()




