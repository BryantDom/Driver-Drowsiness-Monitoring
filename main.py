from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.optim as optim
from utils import apply_prune, apply_channels_prune,print_prune_channel,print_prune,load_data,load_data_adj,set_seed
from torch.utils.data import TensorDataset, DataLoader
from method import basePoint,BDSAG,wavelet_transform
from Network.nowBest import CNN_my
from train import EEGtrain,EEGtest

def main():
    xdata, ydata, subIdx = load_data()

    # BDST
    mean_power = basePoint(xdata)

    # 小波变换
    xdata = wavelet_transform(xdata)

    subjnum = 11
    results1 = np.zeros(subjnum)
    acc = np.zeros(subjnum)
    recall = np.zeros(subjnum)
    f1 = np.zeros(subjnum)
    spe = np.zeros(subjnum)
    pre = np.zeros(subjnum)

    # xdata = BDSAG(xdata, mean_power)
    xdata = load_data_adj('adj_f_data.pkl')

    for i in range(1, subjnum + 1):
        trainindx = np.where(subIdx != i)[0]
        xtrain = xdata[trainindx]
        y_train = ydata[trainindx]
        # form the testing data
        testindx = np.where(subIdx == i)[0]
        xtest = xdata[testindx]
        y_test = ydata[testindx]

        train_ch01 = np.array(xtrain, dtype=float)
        val_ch01 = np.array(xtest, dtype=float)
        train_ch01 = torch.FloatTensor(train_ch01)
        val_ch01 = torch.FloatTensor(val_ch01)
        train_label11 = np.array(y_train, dtype=int)
        val_label11 = np.array(y_test, dtype=int)
        train_label11 = torch.LongTensor(train_label11)
        val_label11 = torch.LongTensor(val_label11)
        train_set = TensorDataset(train_ch01, train_label11)
        val_set = TensorDataset(val_ch01, val_label11)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
        # test_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)
        test_x = val_ch01
        test_y = val_label11

        percent = 0.2
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch')
        parser.add_argument('--percent', type=list,
                            default=[percent, percent, percent, percent, percent, percent, percent, percent,
                                     percent, percent, percent, percent, percent, percent, percent, percent,
                                     percent, percent, percent, percent, percent, percent, percent, percent,
                                     percent, percent, percent, percent, percent, percent, percent, percent,
                                     percent, percent, percent, percent, percent, percent, percent, percent,
                                     percent, percent, percent, percent, percent, percent, percent, percent,
                                     percent, percent, percent, percent, percent, percent],
                            metavar='P', help='pruning percentage (default: 0.2)')
        parser.add_argument('--num_epochs', type=int, default=2, metavar='P',
                            help='number of epochs to pretrain (default: 3)')
        parser.add_argument('--lr', type=float, default=0.0015, metavar='LR',
                            help='learning rate (default: 1e-2)')
        parser.add_argument('--seed', type=int, default=75, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--node_number', type=int, default=384, metavar='N',
                            help='the number of input points (default: 100)')
        args = parser.parse_args()

        # device = torch.device("cuda" if use_cuda else "cpu")
        device = torch.device("cuda")
        set_seed(args)

        model = CNN_my().to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 训练和模型保存
        # acc_11Subject, acc_11Subject_ZU = EEGtrain(args, model, device, train_loader, test_x, test_y, optimizer,
        #                                         dataset, model_style, i)
        # torch.save(model.state_dict(), 'test.pth')

        # 模型读取
        new_dict = torch.load('test.pth')
        model.load_state_dict(new_dict)

        # 模型剪枝
        num_channels_ratio = 0.1
        mask = apply_channels_prune(model=model, device="cuda", num_channels_ratio=num_channels_ratio)
        # print_prune_channel(model)
        mask = apply_prune(model, device, args)
        # print_prune(model)

        acc_11, recall_11, f1_11, spe_11, pre_11 = EEGtest(model, device, test_x, test_y, i)
        acc[i - 1] = acc_11
        recall[i - 1] = recall_11
        f1[i - 1] = f1_11
        spe[i - 1] = spe_11
        pre[i - 1] = pre_11

    print('训练平均准确率:', np.mean(results1))
    print('神经元剪枝', percent,'通道剪枝', num_channels_ratio,'后直接测试的平均准确率:', np.mean(acc))
    print('神经元剪枝', percent,'通道剪枝', num_channels_ratio,'后直接测试的平均recall:', np.mean(recall))
    print('神经元剪枝', percent,'通道剪枝', num_channels_ratio,'后直接测试的平均F1 score:', np.mean(f1))
    print('神经元剪枝', percent,'通道剪枝', num_channels_ratio,'后直接测试的平均spe:', np.mean(spe))
    print('神经元剪枝', percent,'通道剪枝', num_channels_ratio,'后直接测试的平均pre:', np.mean(pre))

if __name__ == "__main__":
    main()
