import numpy as np
import torch
import torch
import torch.nn as nn
import torch.onnx
import torchsummary
import tvm
from tvm import relay
import time
from thop import profile
from tvm.contrib import graph_executor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
from torch.nn import Sequential, Linear, ReLU
import random

class BatchChannelNormalization2d(nn.Module):
    def __init__(self, num_channels, num_groups=4, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNormalization2d, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.momentum = momentum

        # BatchNorm2d with num_groups
        self.batchnorm = nn.BatchNorm2d(num_channels, eps=epsilon, momentum=momentum, affine=False)

    def forward(self, x):
        # Reshape input tensor to have num_groups channels
        x_reshaped = x.view(-1, self.num_groups, self.num_channels // self.num_groups, x.size(2), x.size(3))

        # BatchNorm2d along the num_groups dimension
        x_normalized = self.batchnorm(x_reshaped.contiguous().view(-1, self.num_channels, x.size(2), x.size(3)))

        return x_normalized

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

def subgraph(data, ratio):

    # 获取节点数和特征维度
    num_graphs, num_nodes, feature_dim = data.x.size()

    # 计算子图中的节点数
    sub_num = int(num_nodes * ratio)

    # 将edge_index转为numpy数组
    edge_index = data.edge_index.numpy()

    # 从所有节点中随机选择一个作为子图的起始节点
    unique_nodes = np.unique(edge_index)
    start_node = [np.random.choice(list(unique_nodes), size=1)[0]]
    neighbors = set([n for n in edge_index[1][edge_index[0] == start_node[0]]])

    # 扩展子图，保证选择的节点不在子图中
    while len(start_node) <= sub_num:
        neighbors = neighbors - neighbors.intersection(set(start_node))
        if len(neighbors) == 0:
            break
        sample_node = np.random.choice(list(neighbors))
        if set(start_node) == neighbors:
            break
        if sample_node in start_node:
            continue

        start_node.append(sample_node)
        neighbors = neighbors.union(set([n for n in edge_index[1][edge_index[0] == start_node[-1]]]))

    # 计算要删除和保留的节点索引
    idx_drop = [n for n in range(num_nodes) if not n in start_node]
    idx_nondrop = start_node

    # 将edge_index转为numpy数组
    edge_index = data.edge_index.numpy()

    # 创建邻接矩阵并进行图的剪枝
    adj = torch.eye(num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()
    edge_index = edge_index.numpy()

    # 计算剪枝后的边数
    _, edge_num = edge_index.shape

    # 获取不缺失的节点索引
    idx_not_missing = [n for n in range(num_nodes) if (n in edge_index[0] or n in edge_index[1])]

    # 更新数据的节点特征和边索引
    num_nodes = len(idx_not_missing)
    data.x = data.x[:, idx_not_missing, :]

    # 构建新的节点索引字典
    idx_dict = {idx_not_missing[n]: n for n in range(num_nodes)}

    # 重新映射边索引
    new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                      not edge_index[0, n] == edge_index[1, n]]
    if not new_edge_index:
        new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num)]

    new_edge_index = torch.tensor(new_edge_index)
    new_edge_index.transpose_(0, 1)
    data.edge_index = new_edge_index

    return data

def inputsubdata(x, edge_index, ratio):
    data = Data(x=x, edge_index=edge_index)

    new_data = subgraph(data, ratio)

    subx = new_data.x
    edge_index = new_data.edge_index

    # 确保subx的维度是合适的
    # if subx.dim() == 2:
    #     subx = subx.unsqueeze(0)  # 将维度扩展为三维

    nn = Sequential(Linear(384, 32), ReLU(), Linear(32, 384))
    conv = GINConv(nn).to('cuda')
    subx = subx.to('cuda')
    edge_index = edge_index.to('cuda')

    x1 = conv(subx, edge_index).to('cuda')
    return x1


class CNN_my(torch.nn.Module):

    """
    The codes implement the CNN model proposed in the paper "EEG-based Cross-Subject Driver Drowsiness Recognition
    with an Interpretable Convolutional Neural Network".(doi: 10.1109/TNNLS.2022.3147208)

    The network is designed to classify multi-channel EEG signals for the purposed of driver drowsiness recognition.

    Parameters:

    classes       : number of classes to classify, the default number is 2 corresponding to the 'alert' and 'drowsy' labels.
    sampleChannel : channel number of the input signals.
    sampleLength  : the length of the EEG signals. The default value is 384, which is 3s signal with sampling rate of 128Hz.
    N1            : number of nodes in the first pointwise layer.
    d             : number of kernels for each new signal in the second depthwise layer.
    kernelLength  : length of convolutional kernel in second depthwise layer.

    if you have any problems with the code, please contact Dr. Cui Jian at cuij0006@ntu.edu.sg
    """

    def __init__(self, classes=2, sampleChannel=30, sampleLength=384 ,N1=8, d=2,kernelLength=32):
        super(CNN_my, self).__init__()
        self.pointwise = torch.nn.Conv2d(1,N1,(1,kernelLength))

        self.pconv = Partial_conv3(d*N1*2,d*N1*2//8)
        self.attn = simam_module(N1)
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1)
        self.activ=torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm2d(d*N1*2,track_running_stats=False)
        # self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))
        self.GAP=torch.nn.AvgPool2d((1, kernelLength))

        self.fc = torch.nn.Linear(320, 256)
        self.bn1 = torch.nn.BatchNorm1d(1,track_running_stats=False)
        self.activ1=torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(256, classes)

        self.softmax=torch.nn.LogSoftmax(dim=1)
        self.BCN = BatchChannelNormalization2d(d*N1)
        self.aggregate_weightT = torch.nn.Parameter(torch.ones(1, 1, 161))
        self.edge_index = torch.randint(10, 384, (2, 500))  # Assuming 500 edges for illustration


    def forward(self, inputdata):

        inputdata_local1 = inputsubdata(inputdata, self.edge_index,0.4)
        # inputdata_local2 = inputsubdata(inputdata, self.edge_index,0.3)

        inputdata_global1 = inputsubdata(inputdata, self.edge_index,0.9)
        # inputdata_global2 = inputsubdata(inputdata, self.edge_index,0.7)

        inputdata_gl = torch.cat([inputdata_local1, inputdata_global1, inputdata],dim=1).to('cuda')

        i, j,k = inputdata_gl.shape
        weight_num = torch.nn.Parameter(torch.ones(1, 1, j)).to('cuda')

        inputdata = torch.matmul(weight_num, inputdata_gl).to('cuda')
        inputdata = inputdata.unsqueeze(1)

        intermediate = self.pointwise(inputdata)

        # intermediate = self.pconv(intermediate)
        # intermediate = self.attn(intermediate)
        intermediate1 = self.depthwise(intermediate)

        intermediate = self.pointwise(inputdata)

        # intermediate = self.pconv(intermediate)
        # intermediate = self.attn(intermediate)
        intermediate2 = self.depthwise(intermediate)

        intermediate = torch.concat([intermediate1,intermediate2],dim=1)

        intermediate = self.pconv(intermediate)
        intermediate = self.activ(intermediate)
        intermediate = self.batchnorm(intermediate)
        intermediate = self.GAP(intermediate)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        intermediate = self.fc(intermediate)
        intermediate = intermediate.view(intermediate.size()[0],1, -1)
        intermediate = self.activ1(intermediate)
        intermediate = self.bn1(intermediate)

        intermediate = intermediate.view(intermediate.size()[0], -1)
        intermediate = self.fc1(intermediate)



        x = intermediate

        return F.log_softmax(x, dim=1), x
def set_seed(args):
    torch.manual_seed(args)
    torch.cuda.manual_seed_all(args)
    np.random.seed(args)
    random.seed(args)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(13)

    input_data = torch.randn(32,384,384)
    model = CNN_my()
    result = model(input_data)
    print(result[0].shape)
    import torch


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def compute_flops(model, input_shape):
        # 将模型设置为评估模式
        model.eval()

        # 遍历模型的每一层，统计乘法操作的数量（FLOPs）
        flops = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # 如果是剪枝的卷积层，则根据剪枝后的权重数量计算FLOPs
                flops += abs((module.weight.sum().item()) * module.out_channels * module.kernel_size[0] *
                             module.kernel_size[1] *
                             input_shape[-2] * input_shape[-1] / (module.stride[0] * module.stride[1]))
            elif isinstance(module, torch.nn.Linear):
                # 计算全连接层的FLOPs
                flops += abs((module.weight.sum().item()) * module.out_features)

            elif isinstance(module, Partial_conv3):
                # 计算部分卷积的FLOPs
                flops += abs(
                    module.dim_conv3 * module.dim_conv3 * 3 * 3 * module.dim_conv3 * input_shape[-2] * input_shape[
                        -1])

            # 还需要根据量化操作更新FLOPs的计算方式

        return flops


    # Example usage:
    input_size = (1,384, 384)  # Example input size (batch_size, channels, height, width)
    params_count = count_parameters(model)
    flops_count = compute_flops(model, input_size)

    print("Parameters count:", params_count)
    print("FLOPs count:", flops_count)

    from thop import profile

    with torch.no_grad():

        # 计算 FLOPs 和参数数量
        input_data = (input_data,)
        flops, params = profile(model, input_data)

        print("Total parameters:", params)
        print("Total FLOPs:", flops)
    params_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print("模型大小（量化前）：", params_size / 1024, "字节")