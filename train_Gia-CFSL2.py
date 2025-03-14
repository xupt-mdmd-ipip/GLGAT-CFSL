import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pickle
import random
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import utils
import imp
from model.IDE_block import DPGN
import scipy.io as io
from model import CSA_block
from model.Network import *
from master import zhenshi_OD23D

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join('./config', 'Indian_pines.py'),
                    help='config file with parameters of the experiment. '
                         'It is assumed that the config file is placed under the directory ./config')
args = parser.parse_args()

# Hyper Parameters
config = imp.load_source("", args.config).config
train_opt = config['train_config']
data_path = config['data_path']
save_path = config['save_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']

patch_size = train_opt['patch_size']
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
N_DIMENSION = train_opt['n_dim']
CLASS_NUM = train_opt['class_num']
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
LEARNING_RATE = train_opt['lr']
lambda_1 = train_opt['lambda_1']
lambda_2 = train_opt['lambda_2']
GPU = config['gpu']

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = train_opt['test_class_num']  # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = train_opt['test_lsample_num_per_class']  # the number of labeled samples per class
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# 设置初始化加载信息部分
utils.same_seeds(0)  # 设置相同的随机数


def _init_():
    """
    创建文件夹
    """
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')


_init_()

# load source domain data set
print("before")
print("load source domain data set:")
with open(os.path.join(data_path, source_data), 'rb') as handle:
    source_imdb = pickle.load(handle)
print("print(source_imdb.keys()):", source_imdb.keys())
print("source_imdb['Labels']", source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']  # (77592, 9, 9, 128)
labels_train = source_imdb['Labels']  # 77592,
print('source domain data_train.shape：', data_train.shape)
print('source domain labels_train.shape:', labels_train.shape)

keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print('keys_all_train：', keys_all_train)  # 19个类别[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print('label_encoder_train:', label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print('train_set.keys()：', train_set.keys())  # train_set.keys()：
# dict_keys([5, 0, 8, 13, 7, 6, 4, 17, 10, 1, 3, 12, 16, 14, 2, 9, 11, 15, 18])

data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("\nNum classes for source domain datasets: " + str(len(data)))  # 19个类别
# It contains 19 classes and has 2517 × 2335 pixels.
# Its spatial resolution is 2.5 m per pixel.
# It consists of 128 spectral bands, which range from 363 to 1018 nm.
print(data.keys())  # dict_keys([5, 0, 8, 13, 7, 6, 4, 17, 10, 1, 3, 12, 16, 14, 2, 9, 11, 15, 18])
data = utils.sanity_check(data)  # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
print("\nsource few-shot classification data:")
metatrain_data = data
print("@metatrain_data.key():", len(metatrain_data.keys()), metatrain_data.keys())
# 18 dict_keys([5, 0, 8, 13, 7, 6, 4, 17, 10, 1, 3, 12, 16, 14, 2, 9, 11, 15])
# 这是经过函数筛选之后得到的数据
del data

# source domain adaptation data
print("\nsource domain adaptation data:")
print('source_imdb[data].shape：', source_imdb['data'].shape)  # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print('source_imdb[data].shape：', source_imdb['data'].shape)  # (77592, 9, 9, 100) (9, 9, 128, 77592)
print('source_imdb[Labels]：', source_imdb['Labels'])

# target domain data set
# load target domain data set
test_data = os.path.join(data_path, target_data)
test_label = os.path.join(data_path, target_data_gt)

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# get train_loader and test_loader 获取训练集和测试集部分
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):
    """
    和DCFSL相比，在参数部分多加了HalfWidth，这在原来作为局部变量设置为4
    """
    print('\nData_Band_Scaler.shape：', Data_Band_Scaler.shape)  # (145, 145, 200)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    # HalfWidth
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)  # 取得的样本数，10249

    # Sampling samples抽样样本
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    # 划分出训练集和测试集80 10169
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)  # 函数是打乱序列里面的元素，并随机排列的。
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('\nthe number of train_indices训练集个数:', len(train_indices))  # 80训练的样本数
    print('the number of test_indices测试集个数:', len(test_indices))  # 10169测试的样本数
    print('the number of train_indices after data argumentation:', len(da_train_indices))
    # 3200总共有16个类别，每个类别取出200个样本，共计3200个，同比相对于其他的数据集样本类别只有9个，那么没类取200个，共计取得1800个。
    print('labeled sample indices训练集为:', train_indices)

    # 这是另外的一个部分
    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n) (9, 9, 200, 10249)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)  # (10249,)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)  # (10249,)
    # imdb = {'data': np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
    #                          dtype=np.float32), 'Labels': np.zeros([nTrain + nTest], dtype=np.int64),
    #         'set': np.zeros([nTrain + nTest], dtype=np.int64)}
    print('\nimdb[data].shape, imdb[Labels].shape, imdb[set].shape\nbefore', imdb['data'].shape, imdb['Labels'].shape,
          imdb['set'].shape)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('imdb[data].shape, imdb[Labels].shape, imdb[set].shape\nafter', imdb['data'].shape, imdb['Labels'].shape,
          imdb['set'].shape)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n) (9, 9, 200, 3200)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)  # (3200,)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)  # (3200,)
    print('imdb_da_train[data].shape, imdb_da_train[Labels].shape, imdb_da_train[set].shape\nbefore',
          imdb_da_train['data'].shape, imdb_da_train['Labels'].shape, imdb_da_train['set'].shape)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('imdb_da_train[data].shape, imdb_da_train[Labels].shape, imdb_da_train[set].shape\nafter',
          imdb_da_train['data'].shape, imdb_da_train['Labels'].shape, imdb_da_train['set'].shape)
    # 16类每类取出200个，共计3200个样本
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


# get_target_dataset 获取目标域的数据
def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, patch_size):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,
        class_num=class_num, shot_num_per_class=shot_num_per_class, HalfWidth=patch_size // 2)
    # 16 classes and 5 labeled samples per class
    print("\nbefore")
    print('train_loader', train_loader)
    print('test_loader', test_loader)
    # print('imdb_da_train', imdb_da_train)
    print('G', G.shape)
    print('RandPerm', RandPerm.shape)
    print('Row', Row.shape)
    print('Column', Column.shape)
    print('nTrain', nTrain)

    train_datas, train_labels = train_loader.__iter__().next()
    print('\ntrain labels:\n', train_labels)
    print('size of train datas:', train_datas.shape)  # size of train datas: torch.Size([80, 200, 9, 9])

    print('imdb_da_train.keys()', imdb_da_train.keys())
    print('imdb_da_train[data].shape', imdb_da_train['data'].shape)  # (9, 9, 200, 3200)
    print('imdb_da_train[Labels]', imdb_da_train['Labels'])  # [ 0  0  0 ... 15 15 15]

    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print('target_da_datas.shape:', target_da_datas.shape)  # (9, 9, 200, 3200)->(3200, 200, 9, 9)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)  # [ 0  0  0 ... 15 15 15]

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print('imdb_da_train[data].shape:', imdb_da_train['data'].shape)  # (9, 9, 200, 3200)
    print('imdb_da_train[Labels]:', imdb_da_train['Labels'])  # [ 0  0  0 ... 15 15 15]

    print('\nafter')
    print('train_loader', train_loader)
    print('test_loader', test_loader)
    # print('imdb_da_train', imdb_da_train)
    # print('target_da_metatrain_data:', target_da_metatrain_data)
    # print('target_loader:', target_loader)
    print('G', G.shape)
    print('RandPerm', RandPerm.shape)
    print('Row', Row.shape)
    print('Column', Column.shape)
    print('nTrain', nTrain)
    # 和DCFSL相比少了target_loader这个参数的返回值
    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain


# TODO Network
# ----------------------------------------------------------------------------------------------------
# 从model的Network脚本中加载网络参数
class Network(nn.Module):
    def __init__(self, patch_size, emb_size, N_DIMENSION):
        super(Network, self).__init__()
        # self.feature_encoder = D_Res_3d_CNN(1, 8, 16, patch_size, emb_size)
        self.feature_encoder = zhenshi_OD23D.DYConv_2d(N_DIMENSION, emb_size)  # 100,64
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'):
        if domain == 'target':
            x = self.target_mapping(x)  # (45, 100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (45, 100,9,9)
        feature = self.feature_encoder(x)  # (45, 64)
        return feature


# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# 这里的函数在一轮训练中会初始化使用到
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


crossEntropy = nn.CrossEntropyLoss().to(GPU)
domain_criterion = nn.BCEWithLogitsLoss().to(GPU)


# 欧几里得度量
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------

# run 10 times
nDataSet = 10

# 评价指标
acc = np.zeros([nDataSet, 1])
time_list = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None
class_map = np.zeros((nDataSet,), dtype=np.object)
feature_emb_cell = np.zeros((nDataSet,), dtype=np.object)
test_labels_cell = np.zeros((nDataSet,), dtype=np.object)
test_labels_end, feature_emb_end = [], []

seeds = [1336, 1330, 1220, 1233, 1229, 1236, 1226, 1235, 1337, 1224]
# TODO nDataSet
for iDataSet in range(nDataSet):
    print("\nfor iDataSet in range(nDataSet):")
    print("--------------------------------------------------")
    print('emb_size0:', emb_size)  # 64
    print('num_generation:', config['num_generation'])
    print('patch_size:', patch_size)  # 8
    print('seeds:', seeds[iDataSet])  # seeds = [1336, 1330, 1220, 1233, 1229, 1236, 1226, 1235, 1337, 1224]

    # 一样load target domain data for training and testing 加载目标领域数据集用于训练和测试
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
        shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS, patch_size=patch_size)

    # feature_encoder = Network(patch_size, emb_size)
    feature_encoder = Network(patch_size, emb_size, N_DIMENSION)  # 9, 64

    # 多加 IDE_block
    IDE_block_src = DPGN(config['num_generation'],
                         train_opt['dropout'],
                         CLASS_NUM * SHOT_NUM_PER_CLASS,
                         CLASS_NUM * SHOT_NUM_PER_CLASS + CLASS_NUM * QUERY_NUM_PER_CLASS,
                         config['point_distance_metric'],
                         config['distribution_distance_metric'],
                         emb_size).to(GPU)
    IDE_block_tar = DPGN(config['num_generation'],
                         train_opt['dropout'],
                         CLASS_NUM * SHOT_NUM_PER_CLASS,
                         CLASS_NUM * SHOT_NUM_PER_CLASS + CLASS_NUM * QUERY_NUM_PER_CLASS,
                         config['point_distance_metric'],
                         config['distribution_distance_metric'],
                         emb_size).to(GPU)

    # 多加 CSA_block_att 第一个变量类别数量用了3处，后面三个变量没有用到
    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(CLASS_NUM, SHOT_NUM_PER_CLASS,
                                                                                   QUERY_NUM_PER_CLASS,
                                                                                   train_opt['batch_task'], GPU)
    print("\nnum_supports:", num_supports)  # 支撑集的类别数 16个类别
    print("num_samples:", num_samples)  # 样本数目 16个类别乘以20个扩展样本=320
    print("query_edge_mask:", query_edge_mask)  # (1, 320, 320)
    print("evaluation_mask:", evaluation_mask)  # (1, 320, 320)

    CSA_block_att = CSA_block.AttentionalGNN(num_supports, emb_size, ['cross'] * config['num_generation']).to(GPU)

    # 将以上4个变量进行优化器优化
    IDE_block_src_optim = torch.optim.Adam(IDE_block_src.parameters(), lr=LEARNING_RATE,
                                           weight_decay=train_opt['weight_decay'])
    IDE_block_tar_optim = torch.optim.Adam(IDE_block_tar.parameters(), lr=LEARNING_RATE,
                                           weight_decay=train_opt['weight_decay'])
    CSA_block_att_optim = torch.optim.Adam(CSA_block_att.parameters(), lr=LEARNING_RATE,
                                           weight_decay=train_opt['weight_decay'])
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),
                                             lr=LEARNING_RATE)  # , weight_decay=train_opt['weight_decay'])

    # feature_encoder.apply(weights_init)
    IDE_block_src.apply(weights_init)
    IDE_block_tar.apply(weights_init)
    CSA_block_att.apply(weights_init)
    feature_encoder.to(GPU)

    feature_encoder.train()
    IDE_block_src.train()
    IDE_block_tar.train()
    CSA_block_att.train()

    # 初始化评价指标
    last_accuracy = 0.0
    last_accuracy_gnn = 0.0
    best_episdoe = 0
    train_loss = []
    total_hit_src, total_num_src, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.05

    print("Training...")
    train_start = time.time()

    # TODO 开始进入一次完整的训练
    for episode in range(EPISODE):
        # get few-shot classification samples
        task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
        support_dataloader_src = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
        query_dataloader_src = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)

        # DCFSL中只有上面3行，下面三行是本文补加的
        task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_tar = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train",
                                                            shuffle=False)
        query_dataloader_tar = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                          shuffle=True)

        ''' sample datas扩充成两个域 '''
        supports_src, support_labels_src = support_dataloader_src.__iter__().next()  # (5, 100, 9, 9)
        querys_src, query_labels_src = query_dataloader_src.__iter__().next()  # (75,100,9,9)

        supports_tar, support_labels_tar = support_dataloader_tar.__iter__().next()
        querys_tar, query_labels_tar = query_dataloader_tar.__iter__().next()

        ''' calculate features扩充成两个域 '''
        support_features_src = feature_encoder(supports_src.to(GPU))
        query_features_src = feature_encoder(querys_src.to(GPU))
        support_features_tar = feature_encoder(supports_tar.to(GPU), domain='target')
        query_features_tar = feature_encoder(querys_tar.to(GPU), domain='target')

        ''' initialize nodes and edges for dual graph model额外添加 使用到图神经网络的方法'''
        batch_src = [support_features_src[1].unsqueeze(0).unsqueeze(0), support_labels_src.unsqueeze(0).unsqueeze(0),
                     query_features_src[1].unsqueeze(0).unsqueeze(0), query_labels_src.unsqueeze(0).unsqueeze(0)]
        batch_tar = [support_features_tar[1].unsqueeze(0).unsqueeze(0), support_labels_tar.unsqueeze(0).unsqueeze(0),
                     query_features_tar[1].unsqueeze(0).unsqueeze(0), query_labels_tar.unsqueeze(0).unsqueeze(0)]

        # initialize variables DPGN main函数中前部分定义的 修改这部分
        tensors_src = utils.allocate_tensors()
        for key, tensor in tensors_src.items():
            tensors_src[key] = tensor.to(GPU)

        # initialize nodes and edges for dual graph model 函数：initialize_nodes_edges
        # 第一行中所有的变量都没用在下文中使用，第二行的变量在后面会使用到
        support_data_src, support_label_src, query_data_src, query_label_src, all_data_src, all_label_in_edge_src, \
            node_feature_gd_src, edge_feature_gp_src, edge_feature_gd_src = utils.initialize_nodes_edges(
            batch_src,
            num_supports,
            tensors_src,
            train_opt['batch_task'],
            query_labels_src.shape[0],
            CLASS_NUM,
            GPU)

        # initialize variables DPGN main函数中前部分定义的 修改这部分
        tensors_tar = utils.allocate_tensors()
        for key, tensor in tensors_tar.items():
            tensors_tar[key] = tensor.to(GPU)
        # initialize nodes and edges for dual graph model 函数：initialize_nodes_edges
        # 第一行中所有的变量都没用在下文中使用，第二行的变量在后面会使用到
        support_data_tar, support_label_tar, query_data_tar, query_label_tar, all_data_tar, all_label_in_edge_tar, \
            node_feature_gd_tar, edge_feature_gp_tar, edge_feature_gd_tar = utils.initialize_nodes_edges(
            batch_tar,
            num_supports,
            tensors_tar,
            train_opt['batch_task'],
            query_labels_tar.shape[0],
            CLASS_NUM,
            GPU)

        ''' calculate prototype network扩充成两个域 '''
        if SHOT_NUM_PER_CLASS > 1:
            support_proto_src = [support_features_src[i].reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1) for i in
                                 range(len(support_features_src))]
            support_proto_tar = [support_features_tar[i].reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1) for i in
                                 range(len(support_features_tar))]
        else:
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar

        ''' the IDE-block 这两部分是新添加的DPGN中main的train函数中有 修改这部分 '''
        # use backbone encode image
        second_last_layer_data_src = torch.cat((support_features_src[0], query_features_src[0])).unsqueeze(0)
        last_layer_data_src = torch.cat((support_features_src[1], query_features_src[1])).unsqueeze(0)

        # use backbone encode image
        second_last_layer_data_tar = torch.cat((support_features_tar[0], query_features_tar[0])).unsqueeze(0)
        last_layer_data_tar = torch.cat((support_features_tar[1], query_features_tar[1])).unsqueeze(0)

        # run the DPGN model
        _, _, _, point_nodes_src, distribution_nodes_src = IDE_block_src(second_last_layer_data_src,
                                                                         last_layer_data_src,
                                                                         node_feature_gd_src,
                                                                         edge_feature_gd_src,
                                                                         edge_feature_gp_src)

        # run the DPGN model
        _, _, _, point_nodes_tar, distribution_nodes_tar = IDE_block_tar(second_last_layer_data_tar,
                                                                         last_layer_data_tar,
                                                                         node_feature_gd_tar,
                                                                         edge_feature_gd_tar,
                                                                         edge_feature_gp_tar)

        # the CSA-block for cross_att_loss 这两部分是新添加的
        cross_att_loss = CSA_block_att(point_nodes_src, point_nodes_tar, distribution_nodes_src, distribution_nodes_tar)

        ''' few-shot learning for fsl_loss '''
        logits_src = euclidean_metric(query_features_src[1], support_proto_src[1])
        f_loss_src = crossEntropy(logits_src, query_labels_src.long().to(GPU))

        logits_tar = euclidean_metric(query_features_tar[1], support_proto_tar[1])
        f_loss_tar = crossEntropy(logits_tar, query_labels_tar.long().to(GPU))
        f_loss = f_loss_src + f_loss_tar

        ''' domain alignment for domain_loss
            在DCFSL中计算为domain adaption
            DCFSL中求解的只有几行代码，
            这一部分在这专门设计了一个函数来进行求解
            修改这部分 '''
        ot_ep, wd_ep, gwd_ep = utils.OT(point_nodes_src, point_nodes_tar)
        ot_ed, wd_ed, gwd_ed = utils.OT(distribution_nodes_src, distribution_nodes_tar)

        domain_loss = ot_ep + ot_ed

        # TODO loss
        # loss计算和DCFSL不太一样fsl loss domain loss cross_att loss修改这部分
        loss = f_loss + lambda_1 * domain_loss + lambda_2 * cross_att_loss  # 0.01

        # Update parameters修改这部分
        feature_encoder.zero_grad()
        IDE_block_src_optim.zero_grad()
        IDE_block_tar_optim.zero_grad()
        CSA_block_att_optim.zero_grad()
        loss.backward()
        feature_encoder_optim.step()
        IDE_block_src_optim.step()
        IDE_block_tar_optim.step()
        CSA_block_att_optim.step()

        # 额外扩充成两个域 这一部分分为了源域和目标域
        total_hit_src += torch.sum(torch.argmax(logits_src, dim=1).cpu() == query_labels_src).item()
        total_num_src += querys_src.shape[0]
        total_hit_tar += torch.sum(torch.argmax(logits_tar, dim=1).cpu() == query_labels_tar).item()
        total_num_tar += querys_tar.shape[0]

        if (episode + 1) % 100 == 0:
            train_loss.append(loss.item())
            print('episode {:>3d}:  domain loss: {:6.4f}, fsl loss: {:6.4f}, cross att loss: {:6.4f}, acc_src {:6.4f}, '
                  'acc_tar {:6.4f}, loss: {:6.4f}'.format(
                episode + 1,
                domain_loss.item(),
                f_loss.item(),
                cross_att_loss.item(),  # 额外添加
                total_hit_src / total_num_src,
                total_hit_tar / total_num_tar,  # 扩充成两个域
                loss.item()))
            print('fea_lr： {:6.6f}'.format(feature_encoder_optim.param_groups[0]['lr']))
            print('wd {:6.6f}, gwd: {:6.6f}'.format((wd_ed + wd_ep).item(), (gwd_ed + gwd_ep).item()))

        # 此if判断和DCFSL基本一致
        if (episode + 1) % 500 == 0 or episode == 0:
            # test
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            predict_gnn = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()
            _, train_features = feature_encoder(Variable(train_datas).to(GPU), domain='target')

            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)  # Nearest Neighbor Classification
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
            test_labels_all, feature_emb = [], []  # 额外添加
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                _, test_features = feature_encoder(Variable(test_datas).to(GPU), domain='target')
                feature_emb.append(test_features.cpu().detach().numpy())
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())

                test_labels = test_labels.numpy()
                test_labels_all.append(test_labels)  # 添加
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            print('seeds:', seeds[iDataSet])
            test_end = time.time()

            # Training mode
            feature_encoder.train()
            if test_accuracy > last_accuracy:
                # save networks 从DCFSL中copy
                torch.save(feature_encoder.state_dict(),
                           str("checkpoints/DFSL_feature_encoder_" + "IP_" + str(iDataSet) + "iter_" + str(
                               TEST_LSAMPLE_NUM_PER_CLASS) + "shot.pkl"))
                print("save networks for episode:", episode + 1)

                last_accuracy = test_accuracy
                best_episdoe = episode

                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)  # A代表每个类别的准确率(1,16)
                best_predict_all = predict  # 额外添加(10169,)
                best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain  # 额外添加
                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)  # kappa系数

                feature_emb_end = np.concatenate(feature_emb)  # 额外添加(10169,64)
                test_labels_end = np.concatenate(test_labels_all)  # 额外添加(10169,)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    # 这个if判断是从DCFSL中copy过来的，原文中没有
    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain

    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('iter:{} best episode:[{}], best accuracy_gnn={}'.format(iDataSet, best_episdoe + 1, last_accuracy_gnn))
    print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))  # 额外添加
    print("accuracy list: ", acc)  # 额外添加
    print('***********************************************************************************')
    # 这层循环表达的应该是绘制分类图的函数，理论上这在DCFSLA中是在最终结尾的时候描述的
    # for i in range(len(best_predict_all)):
    #     best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = \
    #         best_predict_all[i] + 1
    # class_map[iDataSet] = best_G[4:-4, 4:-4]
    time_list[iDataSet] = train_end - train_start  # 额外添加
    feature_emb_cell[iDataSet] = feature_emb_end  # 额外添加
    test_labels_cell[iDataSet] = test_labels_end  # 额外添加

# 下面这一小部分没有做任何改动
AA = np.mean(A, 1)

AAMean = np.mean(AA, 0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("accuracy list: ", acc)
print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
print("accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

# ----------------------------------------------------------------------------------------------------
# classification map

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[
                                                                                                        i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = np.array([0, 0, 0]) / 255.
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = np.array([0, 0, 255]) / 255.
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = np.array([0, 255, 0]) / 255.
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = np.array([0, 255, 255]) / 255.
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = np.array([255, 0, 0]) / 255.
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = np.array([255, 0, 255]) / 255.
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = np.array([255, 255, 0]) / 255.
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = np.array([166, 128, 255]) / 255.
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = np.array([166, 90, 255]) / 255.
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = np.array([192, 128, 192]) / 255.
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = np.array([192, 255, 128]) / 255.
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = np.array([128, 255, 166]) / 255.
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = np.array([166, 166, 0]) / 255.
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = np.array([192, 255, 166]) / 255.
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = np.array([0, 0, 128]) / 255.
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = np.array([0, 255, 192]) / 255.
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = np.array([128, 192, 255]) / 255.

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,
                         "classificationMap/IP.png")
# ----------------------------------------------------------------------------------------------------
