import xml.etree.ElementTree as ET
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import json
import torch
import random
import os
import time
import re
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn as nn


class Train_Dataset(Dataset):
    def __init__(self, train_data):
        self.Data = train_data

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        id = list(self.Data.keys())[index]
        return self.Data[id]+[id]
        #[title,question,label,category,is_anonymous,user_name,id]

class Unlabel_Dataset(Dataset):
    def __init__(self, train_data):
        self.Data = train_data

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        id = list(self.Data.keys())[index]
        txt = self.Data[id][1]
        label = self.Data[id][2]
        question_category = self.Data[id][3]
        question_is_anonymous = int(self.Data[id][4] == 'anonymous')
        return txt, label, question_category, question_is_anonymous,id


class elmo_cnn_net(torch.nn.Module):
    def __init__(self, elmo):
        super(elmo_cnn_net, self).__init__()
        Ks = [3, 4, 5]
        embedding_size = 100
        #         self.category_embedding = nn.Embedding(27, embedding_size)
        self.is_anonymous_embedding = nn.Embedding(2, embedding_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, 200, (K, 1024)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.title_transformation = nn.Linear(1024,200)
        self.output_layer = nn.Linear(200+ len(Ks) * 200 + embedding_size, 3)
        self.elmo = elmo

    def process(self, sentence):
        temp = re.split('[ ,.?~!]', sentence)
        return list(filter(lambda x: x != '', temp))[:512]

    def forward(self, input):
        is_anonymous_embeddings = self.is_anonymous_embedding(input[4].to('cuda'))
        title = input[0]
        title = [self.process(i) for i in title]
        sentences = input[1]
        sentences = [self.process(i) for i in sentences]
        title_char_ids = batch_to_ids(title).to('cuda')  # 64*14*50
        sentences_char_ids = batch_to_ids(sentences).to('cuda')
        title_elmo_out = self.elmo(title_char_ids)['elmo_representations'][0]  # 64*14*1024
        sentences_elmo_out = self.elmo(sentences_char_ids)['elmo_representations'][0]
        title_embeddings = title_elmo_out.mean(1)
        title_embeddings = nn.ReLU()(self.title_transformation(title_embeddings))
        # title_embeddings = self.dropout(title_embeddings)
        x = sentences_elmo_out.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        x = torch.cat((x, title_embeddings,is_anonymous_embeddings), 1)
        output_vector = self.output_layer(x)
        return output_vector


def get_label(label, mapping_list):
    if label in mapping_list:
        return mapping_list[label]
    else:
        return -1

def get_one(a):
    return list(a)[0], a[list(a)[0]]

def get_data(xml_path, mode='train'):
    data = {}
    print('parsing...', xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    max_length = 0
    for thread in root:
        question_tag = thread[0]
        question_id = question_tag.attrib['RELQ_ID']
        question_category = question_tag.attrib['RELQ_CATEGORY']
        if question_category in CATEGORY_MAPPING:
            question_category = CATEGORY_MAPPING[question_category]
        else:
            question_category = CATEGORY_MAPPING['unseen']
        user_name = question_tag.attrib['RELQ_USERNAME']
        if user_name == 'anonymous':
            question_is_anonymous = 1
        else:
            question_is_anonymous = 0
        if user_name not in USER_NAME_MAPPING:
            user_name = USER_NAME_MAPPING['anonymous']
        else:
            user_name = USER_NAME_MAPPING[user_name]
        if mode != 'test':
            question_fact_label = question_tag.attrib['RELQ_FACT_LABEL']
            label = get_label(question_fact_label, QUESTION_LABELS_MAPPING)
        else:
            label = 0
        question_body = question_tag.find('RelQBody').text
        if not question_body: question_body = ''
        question_subject = question_tag.find('RelQSubject').text
        question_text = question_subject + '. ' + question_body
        # if len(question_text.split()) > max_length: max_length = len(question_text)
        # if label > -1:
        content = [question_subject, question_body, label, question_category, question_is_anonymous,user_name]
        data[question_id] = content
    return data

USING_UNLABEL = False
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2}
with open('data/unlabel_category.json','r') as file:
    CATEGORY_MAPPING = json.load(file)
cwd = os.getcwd().split('question_classification')[0]

train_path = cwd + 'train_and_dev_sets_questions_and_an/questions_train.xml'
dev_path = cwd + 'semeval2018_task8_dev_data/questions_dev.xml'
te_path = cwd + 'semeval2018_task8_evaluation_data/questions_test.xml'

# 格式：{id:[title,question,label,category,is_anonymous,user_name]}
with open('data/unlabel_data.json','r') as file:
    unlabel_data = json.load(file)
with open('data/unlabel_user_name.json','r') as file:
    USER_NAME_MAPPING = json.load(file)

train_data = get_data(train_path)
if USING_UNLABEL:
    with open('prediction/predict_questions_2_0.745_0.0124_selected/soft_label_predict.json') as file:
        soft_label_predict = json.load(file)
    add_data = {}
    num = 0
    soft_label_keys = list(soft_label_predict.keys())
    random.shuffle(soft_label_keys)
    for i in soft_label_keys:
        if max(soft_label_predict[i]) > 0.95:
            add_data[i] = unlabel_data[i]
            num += 1
            add_data[i][2] = np.argmax(soft_label_predict[i])
        if num >= 300:
            break
    print('add %d entrys'%len(add_data))
    train_data.update(add_data)
dev_data = get_data(dev_path)
te_data = get_data(te_path,'test')

dataset = Train_Dataset(train_data)
dev_dataset = Train_Dataset(dev_data)
te_dataset = Train_Dataset(te_data)
unlabel_dataset = Train_Dataset(unlabel_data)

batch_size = 64
train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)
dev_loader = DataLoader(dev_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
te_loader = DataLoader(te_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
unlabel_loader = DataLoader(unlabel_dataset,batch_size=batch_size,shuffle=False,num_workers=1)

options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0)
model = elmo_cnn_net(elmo).cuda()
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999),weight_decay=0.013)

total_loss,part_loss,echo_loss = 0,0,0
print_frequence = 5
max_acc = 0
min_loss = 1000000
#train_batch:{question_title,question,label,category,is_anonymous,id}
for epoch in range(150):
    start_test = False
    time1 = time.time()
    part_loss = 0
    print('epoch:%d'%(epoch+1))
    model.train()
    train_sample,train_total_sample = 0,0
    for index,train_batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(train_batch)
        output = output.squeeze(0)
        temp = torch.argmax(output,1) == train_batch[2].cuda()
        train_sample += temp.sum().tolist()
        train_total_sample += temp.shape[0]
        loss_value = loss(output,train_batch[2].cuda())
        loss_value.backward()
        optimizer.step()
        total_loss += float(loss_value)
        part_loss += float(loss_value)
        echo_loss += float(loss_value)
        if index % print_frequence == print_frequence-1:
            print('%.2f%%,train_loss:%.4f'% (100.0*batch_size*index/len(dataset),part_loss/print_frequence))
            part_loss = 0
    print(echo_loss/(index+1))
    print('train acc:%.3f'%(train_sample/train_total_sample))
    print('train_time:%.4f'%(time.time()-time1))
    true_sample,total_sample = 0,0

    dev_loss = 0
    model.eval()
    part_true_sample = 0
    part_all_sample = 0
    for dev_index,dev_batch in enumerate(dev_loader):
        output = model(dev_batch)
        output = output.squeeze(0)

        mask = output.softmax(1).max(1)[0]>0.9
        output_part = output.masked_select(mask.expand(3,mask.shape[0]).t()).reshape(-1,3)
        if output_part.shape[0]:
            label_part = dev_batch[2].cuda().masked_select(mask)
            temp_part = output_part.argmax(1) == label_part
            part_true_sample += temp_part.sum().tolist()
            part_all_sample += temp_part.shape[0]
        else:
            part_all_sample = 1

        temp = torch.argmax(output,1) == dev_batch[2].cuda()
        true_sample += temp.sum().tolist()
        total_sample += temp.shape[0]
        loss_value = loss(output,dev_batch[2].cuda())
        dev_loss += float(loss_value)
    if max_acc<1.0*true_sample/total_sample:
        max_acc = 1.0*true_sample/total_sample
        if 1.0*true_sample/total_sample > 0.72:
            start_test = True
    if dev_loss<min_loss:
        min_loss = dev_loss
        if min_loss < 0.0116*total_sample and 1.0*true_sample/total_sample >0.68:
            start_test = True
        elif min_loss<0.0113*total_sample and 1.0*true_sample/total_sample >0.67:
            start_test = True
        elif min_loss<0.0125*total_sample and 1.0*true_sample/total_sample >0.688:
            start_test = True
    if start_test:
        prediction_list = []
        print('testing2...')
        for te_index, te_batch in enumerate(te_loader):
            output = model(te_batch)
            result = output.argmax(1)
            prediction_list += list(zip(te_batch[6], result.tolist()))
        fold_name = 'predict_questions_2_' + '%.3f' % (1.0*true_sample/total_sample)+'_%.4f'%(dev_loss/total_sample)
        while os.path.exists('prediction/' + fold_name):
            fold_name += '_1'
        os.mkdir('prediction/' + fold_name)
        with open('prediction/' + fold_name + '/predict_questions.txt', 'w') as f:
            for i, j in prediction_list:
                content = i + '\t' + str(j) + '\n'
                f.write(content)
        print('test over2')
        continue
        print('label data...')
        soft_label_dict = {}
        for unlabel_index, unlabel_batch in enumerate(unlabel_loader):
            output = model(unlabel_batch)
            output = output - output.mean(1).unsqueeze(1).expand_as(output)
            soft_label_dict.update(dict(zip(unlabel_batch[6],output.cpu().data.numpy().tolist())))
        with open('prediction/' + fold_name + '/soft_label_predict_new.json', 'w') as f:
            json.dump(soft_label_dict, f)
    print('part_true_sample:%d, part_all_sample:%d, acc:%.3f'%(part_true_sample,part_all_sample,1.0*part_true_sample/part_all_sample))
    print('true_sample:%d, total_sample:%d, acc:%.2f'%(true_sample,total_sample,1.0*true_sample/total_sample))
    print('dev_loss:%.4f'%(dev_loss/total_sample))
    echo_loss = 0

print(max_acc)