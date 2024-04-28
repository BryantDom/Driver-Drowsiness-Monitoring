import pickle
import random

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

def load_data_adj(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def regularized_nll_loss(args, model, output, target):
    index = 0
    loss = F.nll_loss(output, target)
    return loss

def prune_weight(weight, device, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100*percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask


def prune_sum_channels(weight, device, num_channels_to_prune):
    channel = weight.shape[0]
    num_channels_to_prune = int(num_channels_to_prune * channel)
    weight_numpy = weight.detach().cpu().numpy()

    # Calculate the L2 norm of weights across channels
    # channel_sum_weight = (weight_numpy ** 2).sum(axis=(1, 2, 3)) ** 0.5
    channel_sum_weight = abs(weight_numpy).sum(axis=(1, 2, 3))

    # Find the indices of the smallest 'num_channels_to_prune' L2 norms
    indices_to_prune = channel_sum_weight.argsort()[:num_channels_to_prune]

    # Set the weights of these channels to zero
    weight_numpy[indices_to_prune, :, :, :] = 0

    mask = torch.ones_like(weight).to(device)
    mask[indices_to_prune, :, :, :] = 0
    return mask


def apply_channels_prune(model, device, num_channels_ratio):
    dict_mask = {}
    idx = 0
    # print(f"Pruning lowest {num_channels_ratio*100}% of channels")
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[0] != "fc" and name.split('.')[0] != "batchnorm" and name.split('.')[0] != "bn1" and name.split('.')[0] != "fc1":
            mask = prune_sum_channels(param, device, num_channels_ratio)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask

def apply_prune(model, device, args):
    # returns dictionary of non_zero_values' indices
    # print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_weight(param, device, args.percent[idx])
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))

def print_prune_channel(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[0] != "fc" and name.split('.')[0] != "batchnorm" and \
                name.split('.')[0] != "bn1" and name.split('.')[0] != "fc1":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))

def load_data():
    filename = r'data/dataset.mat'
    tmp = sio.loadmat(filename)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])
    subIdx = np.array(tmp['subindex'])
    label.astype(int)
    subIdx.astype(int)
    samplenum = label.shape[0]
    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = label[i]
    #   only channel 28 is used, which corresponds to the Oz channel
    selectedchan = [28]
    #   update the xdata and channel number
    xdata = xdata[:, selectedchan, :]
    channelnum = len(selectedchan)
    xdata = np.squeeze(xdata, axis=1)

    return xdata, ydata, subIdx

















