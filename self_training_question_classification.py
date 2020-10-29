import xml.etree.ElementTree as ET
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import json
import torch
import os
import argparse
import math
import time
import re
from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
import torch.nn as nn

class Train_Dataset(Dataset):
    def __init__(self, train_data):
        self.Data = train_data.contains

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        question = self.Data[index]
        return [question.subject,question.body,question.num_label,
                question.num_category,question.is_anonymous,question.user_name,
                question.id,USER_POOL.id_to_pools[question.user_id].upp,
                USER_POOL.id_to_pools[question.user_id].polarity_1,
                USER_POOL.id_to_pools[question.user_id].polarity_2]
        # return self.Data[id]+[id]
        #[subject,question,label,category,is_anonymous,user_name,id]

class Question():
    def __init__(self):
        self.prediction = np.zeros(3)
        pass

class User_pool():
    def __init__(self):
        self.id_to_pools = {}
        self.user_id_pools = {}

    def __contains__(self, item):
        return item in self.user_id_pools

    def __len__(self):
        return len(self.user_id_pools)

    def _compute(self):
        self.valid_pools = {}
        for user_id in self.id_to_pools:
            if self.id_to_pools[user_id].user_name == 'anonymous':
                continue
            if len(self.id_to_pools[user_id].questions)<4:
                continue
            self.valid_pools[user_id] = self.id_to_pools[user_id]
        delete_key = []
        valid_key = []
        for user in self.valid_pools.values():
            user.predictions = []
            for question in user.questions:
                if question.dataset_flag != 'test':
                    user.predictions.append(question.prediction)
            if len(user.predictions)<4:
                delete_key.append(user.user_id)
                continue
            valid_key.append(user.user_id)
            user.aver_predictions = np.average(user.predictions,0)
            user.max_aver_predictions = np.max(user.aver_predictions)
            user.max_index = np.argmax(user.aver_predictions)
            user.defined_value = self.defined_rule(user,0.5)
        for key in delete_key:
            self.valid_pools.pop(key)
        for question_index in range(len(unlabel_data.contains)-1,-1,-1):
            if unlabel_data.contains[question_index].user_id not in valid_key:
                unlabel_data.contains.pop(question_index)

    def compute_upp(self):
        for u in self.id_to_pools:
            prediction = [i.prediction for i in self.id_to_pools[u].questions if i.label != 'nolabel']
            if not prediction:
                self.id_to_pools[u].polarity = 0
                self.id_to_pools[u].upp = np.zeros(3)
                continue
            self.id_to_pools[u].upp = np.average(prediction,0)
            self.id_to_pools[u].polarity_1 = self.compute_polarity(prediction,0.5,'MUPP')
            self.id_to_pools[u].polarity_2 = self.compute_polarity(prediction,0.5,'EUPP')

    def compute_polarity(self,prediction,alpha,mode):
        part2 = 1 - alpha / len(prediction)
        upp = np.average(prediction,0)
        if mode == 'MUPP':
            part1 = upp.max()
        if mode == 'EUPP':
            temp = -np.log(upp)
            temp[temp == float('inf')] = 0
            H = np.sum(temp * upp)
            part1 = 1 - H / np.log(3)
        return part1*part2

    def defined_rule(self,user,alpha):
        temp = -np.log(user.aver_predictions)
        temp[temp==float('inf')] = 0
        H = np.sum(temp*user.aver_predictions)
        part1 = 1 - H/np.log(3)
        part2 = 1 - alpha/len(user.predictions)
        return part1*part2

    def subsample(self,sorted_user,mode,user_count,power=8):
        if mode == 'defined_value':
            norm = sum([math.pow(t.defined_value, power) for t in sorted_user])  # Normalizing constant
        elif mode =='max_aver_predictions':
            norm = sum([math.pow(t.max_aver_predictions,power) for t in sorted_user])
        table_size = int(1e8)  # Length of the unigram table
        table = []
        for j,user_item in enumerate(sorted_user):
            if mode == 'defined_value':
                p = round(float(math.pow(user_item.defined_value, power))/norm * table_size)
            elif mode == 'max_aver_predictions':
                p = round(float(math.pow(user_item.max_aver_predictions, power))/norm * table_size)
            table += [j] * p
        indices = np.random.randint(low=0, high=len(table), size=user_count)
        return [sorted_user[table[i]] for i in indices]

    def sort_and_pull_items(self,num,threshold,mode='max_aver_predictions',subsample=False,user_num=50,select_user=False):
        if mode == 'max_aver_predictions':
            sorted_user = sorted(self.valid_pools.values(),key=lambda a:(a.max_aver_predictions,len(a.predictions)),reverse=True)
        elif mode == 'defined_value':
            sorted_user = sorted(self.valid_pools.values(), key=lambda a: (a.defined_value),reverse=True)
        if subsample:
            sorted_user = self.subsample(sorted_user[:1000],mode,user_num)
            num = 100000
        if select_user:
            sorted_user = sorted_user[:50]
            num = 100000
        pull_lists = []
        num_limit = num/3
        buffer = num/6
        index = [0,0,0]
        for user in sorted_user:
            if len(user.questions) < 5:
                continue
            if index[user.max_index] >=num_limit + buffer:
                continue
            for question in user.questions:
                if index[user.max_index] >= num_limit + buffer:
                    break
                if question.dataset_flag == 'unlabel':
                    if question.prediction[user.max_index] >threshold:
                        if sum(index)>=num:
                            return pull_lists
                        pull_lists.append(question)
                        question.old_label = question.num_label
                        question.num_label = user.max_index
                        index[user.max_index] += 1
        print('not enough items, only get %d items' %len(pull_lists))
        return pull_lists


class User():
    def __init__(self,question):
        self.user_id = question.user_id
        self.user_name = question.user_name
        self.questions = [question]

class Mydataset():
    def __init__(self):
        self.contains = []
        self.id_to_question = {}
    def __len__(self):
        return len(self.contains)
    def _process(self):
        QUESTION_LABELS_MAPPING = {'Opinion': 0, 'Factual': 1, 'Socializing': 2,'nolabel':3}
        with open('data/unlabel_category.json', 'r') as file:
            CATEGORY_MAPPING = json.load(file)
        for question in self.contains:
            self.id_to_question[question.id] = question
            question.num_label = QUESTION_LABELS_MAPPING[question.label]
            if question.num_label<3:question.prediction[question.num_label] = 1
            question.is_anonymous = question.user_name == 'anonymous'
            question.num_category = CATEGORY_MAPPING[question.category]
    def _add_prediction(self,soft_label_prediction):
        for question in self.contains:
            predict = soft_label_prediction[question.id]
            question.prediction = predict
            question.label = 'soft_label'
    def _extend(self,add_data):
        tmp = add_data[:]
        tmp.extend(self.contains)
        self.contains = tmp
        for question in self.contains:
            self.id_to_question[question.id] = question

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
        if USING_CUDA:
            is_anonymous_embeddings = self.is_anonymous_embedding(input[4].to('cuda'))
        else:
            is_anonymous_embeddings = self.is_anonymous_embedding(input[4])
        title = input[0]
        title = [self.process(i) for i in title]
        sentences = input[1]
        sentences = [self.process(i) for i in sentences]
        if USING_CUDA:
            title_char_ids = batch_to_ids(title).to('cuda')  # 64*14*50
            sentences_char_ids = batch_to_ids(sentences).to('cuda')
        else:
            title_char_ids = batch_to_ids(title)  # 64*14*50
            sentences_char_ids = batch_to_ids(sentences)
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

def get_data(xml_path,USER_POOL,mode='train'):
    dataset = Mydataset()
    print('parsing...', xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for thread in root:
        question = Question()
        question.dataset_flag = mode
        question_tag = thread[0]
        question.id = question_tag.attrib['RELQ_ID']
        question.category = question_tag.attrib['RELQ_CATEGORY']
        question.user_name = question_tag.attrib['RELQ_USERNAME']
        question.user_id = question_tag.attrib['RELQ_USERID']
        if mode == 'train' or mode == 'dev' or mode == 'test':
            question.label = question_tag.attrib['RELQ_FACT_LABEL']
        else:
            question.label = 'nolabel'
        question.body = question_tag.find('RelQBody').text
        if not question.body: question.body = ''
        question.subject = question_tag.find('RelQSubject').text
        if not question.subject:question.subject = ''
        question.text = question.subject + '. ' + question.body
        if question.user_id not in USER_POOL:
            user = User(question)
            USER_POOL.id_to_pools[question.user_id] = user
            USER_POOL.user_id_pools[question.user_id] = len(USER_POOL)
        else:
            USER_POOL.id_to_pools[question.user_id].questions.append(question)
        dataset.contains.append(question)
    dataset._process()
    return dataset

def compute_TP_FP_FN(prediction,label,tp,fp,fn):
    for i in range(label.shape[0]):
        if prediction[i] == label[i]:
            tp[label[i].tolist()] += 1
        if prediction[i] != label[i]:
            fp[prediction[i].tolist()] += 1
            fn[label[i].tolist()] += 1
    return tp,fp,fn

def computer_f1(TP,FP,FN):
    total_prediction = (1.0 * sum(TP))/(sum(TP)+sum(FP))
    total_recall = (1.0*sum(TP))/(sum(TP)+sum(FN))
    mirco_f1 = 2*total_prediction*total_recall/(total_prediction+total_recall)
    macro_f1 = 0
    for i in range(3):
        if TP[i] == 0:
            f1 = 0
        else:
            prediction = (1.0*TP[i])/(TP[i]+FP[i])
            recall = (1.0*TP[i])/(TP[i]+FN[i])
            f1 = 2*prediction*recall/(prediction+recall)
        macro_f1 += f1
    return mirco_f1,macro_f1/3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-unlabel', help='1 for using unlabeled data, 0 for not using unlabeded data', dest='USING_UNLABEL', default=0, type=int)
    parser.add_argument('-u', help='1 for SC-u, 0 for SC', dest='regularization',default=0, type=int)
    parser.add_argument('-polarity', help='1 for EUPP, 0 for MUPP', dest='polarity', default=0, type=int)
    parser.add_argument('-SC_u', help='1 for SC_u, 0 for SC', dest='SC_u', default=0, type=int)


    parser.add_argument('-train', help='Training file', dest='fi', required=False)
    parser.add_argument('-model', help='Output model file', dest='fo', required=False)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=10, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=40, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    parser.add_argument('-iter', help='iter num', dest='iter', default=5, type=int)
    parser.add_argument('-batch', help='batch_size', dest='batch', default=128, type=int)
    parser.add_argument('-mode', help='mode', dest='mode',default=3, type = int)

    #TO DO: parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()
    iter_num = args.iter
    fi = args.fi
    fo = args.fo

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    USING_UNLABEL = args.USING_UNLABEL
    USING_UNLABEL = 1
    SC_u = args.SC_u
    USING_CUDA = True
    cwd = os.getcwd().split('question_classification')[0]
    USER_POOL = User_pool()
    train_path = cwd + 'train_and_dev_sets_questions_and_an/questions_train.xml'
    dev_path = cwd + 'semeval2018_task8_dev_data/questions_dev.xml'
    te_path = cwd + 'factcheck-cqa/data/questions_test.xml'
    unlabel_path = cwd + 'QL-unannotated-data-subtaskA.xml'

    # train: 563(50.3),311(27.8),244(21.8) dev:126(52.7),62(25.9),51(21.4)
    train_data = get_data(train_path, USER_POOL, 'train')
    dev_data = get_data(dev_path, USER_POOL, 'dev')
    te_data = get_data(te_path, USER_POOL, 'test')

    rule = 'baseline'
    subsample = False
    batch_size = 64
    if USING_UNLABEL:
        unlabel_data = get_data(unlabel_path, USER_POOL, 'unlabel')
        print('unlabel data loaded')

        with open('prediction/predict_questions_2_0.745_0.0124_selected/soft_label_predict.json') as file:
            soft_label_prediction = json.load(file)

        unlabel_data._add_prediction(soft_label_prediction)
        USER_POOL._compute()
        # defined_value max_aver_predictions
        rule = 'max_aver_predictions'
        train_length = len(train_data)
        add_data = USER_POOL.sort_and_pull_items(500, 0.8, rule, subsample)
        print('train_data:%d,add %d peso_items to train_data ' % (len(train_data), len(add_data)))
        train_data._extend(add_data)
        unlabel_dataset = Train_Dataset(unlabel_data)
        unlabel_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    USER_POOL.compute_upp()
    dataset = Train_Dataset(train_data) #1118
    dev_dataset = Train_Dataset(dev_data) #239
    te_dataset = Train_Dataset(te_data) #953

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    if USING_CUDA:
        model = elmo_cnn_net(elmo).cuda()
    else:
        model = elmo_cnn_net(elmo)
    loss = nn.CrossEntropyLoss()
    MSEloss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.013)

    total_loss, part_loss, echo_loss = 0, 0, 0
    print_frequence = 5
    max_acc = 0
    min_loss = 1000000
    best_dev_result = [[0,0,0]]*3
    best_te_result = [[0,0,0]]*3
    # train_batch:{question_title,question,label,category,is_anonymous,id}
    for epoch in range(111150):
        if subsample:
            if epoch != 0 and epoch % 3 == 0:  # 每两轮采样
                add_data = USER_POOL.sort_and_pull_items(1000, 0.8, rule, subsample)
                train_data.contains = train_data.contains[:train_length]
                print('%dth train_data:%d,add %d peso_items to train_data ' % (epoch, len(train_data), len(add_data)))
                train_data.contains.extend(add_data)
                dataset = Train_Dataset(train_data)
                train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        start_test = False
        time1 = time.time()
        part_loss = 0
        print('epoch:%d' % (epoch + 1))
        model.train()
        train_sample, train_total_sample = 0, 0
        for index, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(train_batch)
            output = output.squeeze(0)
            if USING_CUDA:
                temp = torch.argmax(output, 1) == train_batch[2].cuda()
            else:
                temp = torch.argmax(output, 1) == train_batch[2]
            train_sample += temp.sum().tolist()
            train_total_sample += temp.shape[0]
            if USING_CUDA:
                loss_value_1 = loss(output, train_batch[2].cuda())
            else:
                loss_value_1 = loss(output, train_batch[2])
            if SC_u:
                l2loss = MSEloss(output,train_batch[7].float().cuda()).sum(1)
                mask = train_batch[9]>1.1
                l2loss_value = (l2loss*mask.float().cuda()).sum()
            else:l2loss_value = 0
            loss_value = loss_value_1 + 0.1 * l2loss_value
            loss_value.backward()
            optimizer.step()
            total_loss += float(loss_value)
            part_loss += float(loss_value)
            echo_loss += float(loss_value)
            if index % print_frequence == print_frequence - 1:
                print(
                    '%.2f%%,train_loss:%.4f' % (100.0 * batch_size * index / len(dataset), part_loss / print_frequence))
                part_loss = 0
        print(echo_loss / (index + 1))
        print('train acc:%.3f' % (train_sample / train_total_sample))
        print('train_time:%.4f' % (time.time() - time1))
        true_sample, total_sample = 0, 0

        dev_loss = 0
        model.eval()
        part_true_sample = 0
        part_all_sample = 0
        TP = [0,0,0]
        FP = [0,0,0]
        FN = [0,0,0]
        for dev_index, dev_batch in enumerate(dev_loader):
            output = model(dev_batch)
            output = output.squeeze(0)

            mask = output.softmax(1).max(1)[0] > 0.9
            output_part = output.masked_select(mask.expand(3, mask.shape[0]).t()).reshape(-1, 3)
            if output_part.shape[0]:
                if USING_CUDA:
                    label_part = dev_batch[2].cuda().masked_select(mask)
                else:
                    label_part = dev_batch[2].masked_select(mask)
                temp_part = output_part.argmax(1) == label_part
                part_true_sample += temp_part.sum().tolist()
                part_all_sample += temp_part.shape[0]
            else:
                part_all_sample = 1
            if USING_CUDA:
                temp = torch.argmax(output, 1) == dev_batch[2].cuda()
                TP,FP,FN = compute_TP_FP_FN(torch.argmax(output, 1),dev_batch[2].cuda(),TP,FP,FN)
            else:
                temp = torch.argmax(output, 1) == dev_batch[2]
            true_sample += temp.sum().tolist()
            total_sample += temp.shape[0]
            if USING_CUDA:
                loss_value = loss(output, dev_batch[2].cuda())
            else:
                loss_value = loss(output, dev_batch[2])
            dev_loss += float(loss_value)
        mirco_f1, macro_f1 = computer_f1(TP, FP, FN)
        print('part_true_sample:%d, part_all_sample:%d, acc:%.3f' % (
            part_true_sample, part_all_sample, 1.0 * part_true_sample / part_all_sample))
        # num_category = [126, 62, 51]
        print(
            'true_sample:%d, total_sample:%d, acc:%.2f, mirco_f1:%.3f,macro_f1:%.3f' % (
            true_sample, total_sample, 1.0 * true_sample / total_sample, mirco_f1, macro_f1))
        print('dev_loss:%.4f' % (dev_loss / total_sample))
        if 1.0 * true_sample / total_sample>best_dev_result[0][0]:
            best_dev_result[0]=[1.0 * true_sample / total_sample,mirco_f1,macro_f1]
        if mirco_f1>best_dev_result[1][1]:
            best_dev_result[1] = [1.0 * true_sample / total_sample,mirco_f1,macro_f1]
        if macro_f1>best_dev_result[2][2]:
            best_dev_result[2] = [1.0 * true_sample / total_sample,mirco_f1,macro_f1]
        print(best_dev_result[0])
        print(best_dev_result[1])
        print(best_dev_result[2])
        if epoch < 10: start_test = True
        if subsample and epoch < 15: start_test = True
        if max_acc < 1.0 * true_sample / total_sample:
            max_acc = 1.0 * true_sample / total_sample
            if 1.0 * true_sample / total_sample > 0.72:
                start_test = True
        if dev_loss < min_loss and dev_loss < 0.0131 * total_sample:
            start_test = True
        if dev_loss < min_loss:
            min_loss = dev_loss
            if min_loss < 0.0116 * total_sample and 1.0 * true_sample / total_sample > 0.68:
                start_test = True
            elif min_loss < 0.0113 * total_sample and 1.0 * true_sample / total_sample > 0.67:
                start_test = True
            elif min_loss < 0.0125 * total_sample and 1.0 * true_sample / total_sample > 0.688:
                start_test = True
        start_test = True
        if start_test:
            true_sample, total_sample = 0, 0
            TP = [0, 0, 0]
            FP = [0, 0, 0]
            FN = [0, 0, 0]
            prediction_list = []
            print('testing2...')
            for te_index, te_batch in enumerate(te_loader):
                output = model(te_batch)
                output = output.squeeze(0)
                if USING_CUDA:
                    temp = torch.argmax(output, 1) == te_batch[2].cuda()
                    TP, FP, FN = compute_TP_FP_FN(torch.argmax(output, 1), te_batch[2].cuda(), TP, FP, FN)
                else:
                    temp = torch.argmax(output, 1) == dev_batch[2]
                true_sample += temp.sum().tolist()
                total_sample += temp.shape[0]
            mirco_f1, macro_f1 = computer_f1(TP, FP, FN)
            num_category = [126, 62, 51]
            print(
                'true_sample:%d, total_sample:%d, acc:%.2f, mirco_f1:%.3f,macro_f1:%.3f' % (
                    true_sample, total_sample, 1.0 * true_sample / total_sample, mirco_f1, macro_f1))
            print('te_loss:%.4f' % (dev_loss / total_sample))
            if 1.0 * true_sample / total_sample > best_te_result[0][0]:
                best_te_result[0] = [1.0 * true_sample / total_sample, mirco_f1, macro_f1]
            if mirco_f1 > best_te_result[1][1]:
                best_te_result[1] = [1.0 * true_sample / total_sample, mirco_f1, macro_f1]
            if macro_f1 > best_te_result[2][2]:
                best_te_result[2] = [1.0 * true_sample / total_sample, mirco_f1, macro_f1]
            print(best_te_result[0])
            print(best_te_result[1])
            print(best_te_result[2])
                # result = output.argmax(1)
                # prediction_list += list(zip(te_batch[6], result.tolist()))
            # if not subsample:
            #     fold_name = 'predict_questions_' + str(epoch) + '_' + rule + '_%.3f' % (
            #     1.0 * true_sample / total_sample) + '_%.4f' % (dev_loss / total_sample)
            # if subsample:
            #     fold_name = 'predict_questions_' + str(epoch) + '_' + rule + '_subsample' + '_%.3f' % (
            #     1.0 * true_sample / total_sample) + '_%.4f' % (dev_loss / total_sample)
            # while os.path.exists('prediction/' + fold_name):
            #     fold_name += '_1'
            # os.mkdir('prediction/' + fold_name)
            # with open('prediction/' + fold_name + '/predict_questions.txt', 'w') as f:
            #     for i, j in prediction_list:
            #         content = i + '\t' + str(j) + '\n'
            #         f.write(content)
            print('test over2')
            continue
            print('label data...')
            soft_label_dict = {}
            for unlabel_index, unlabel_batch in enumerate(unlabel_loader):
                output = model(unlabel_batch)
                output = output - output.mean(1).unsqueeze(1).expand_as(output)
                soft_label_dict.update(dict(zip(unlabel_batch[6], output.cpu().data.numpy().tolist())))
            with open('prediction/' + fold_name + '/soft_label_predict_new.json', 'w') as f:
                json.dump(soft_label_dict, f)

        echo_loss = 0

    print(max_acc)