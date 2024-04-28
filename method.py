import numpy as np
import pywt
from mne.time_frequency import psd_array_multitaper
from scipy.integrate import simps


def BDSAG(data, mean_of_values):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    field = 10
    middle_point = np.zeros(length)
    for i in range(length-field):
        mean_of_values_middle = 0
        for j in range(i,i+field):
            mean_of_values_middle += mean_of_values[j]
        middle_point[i] = mean_of_values_middle/field

    for i in range(length-field,length):
        mean_of_values_middle = 0
        for j in range(length-field,length):
            mean_of_values_middle += mean_of_values[j]
        middle_point[i] = mean_of_values_middle/field

    for k in range(samples):
        print(k)
        for i in range(length):
            for j in range(length):
                adjmatrix[k][i][j] = (data[k][i] - middle_point[j])
                adjmatrix[k][j][i] = (data[k][i] - middle_point[j])
                if i ==j:
                    adjmatrix[k][i][j] = 1
                    adjmatrix[k][j][i] = 1
        diag_values = np.sqrt(1 / field)
        diag_matrix = np.diag(np.full(384, diag_values))
        # 取第一个矩阵并乘以对角矩阵
        adjmatrix[k] = np.matmul(adjmatrix[k], diag_matrix)
    return adjmatrix



def basePoint(data):
    deltapower = np.zeros(384)
    thetapower = np.zeros(384)
    alphapower = np.zeros(384)
    betapower = np.zeros(384)
    mean_power = np.zeros(384)

    for i in range(384):
        psd, freqs = psd_array_multitaper(data[:, i], 128, adaptive=True, normalization='full', verbose=0)
        freq_res = freqs[1] - freqs[0]

        totalpower = simps(psd, dx=freq_res)

        if totalpower < 0.00000001:
            deltapower[i] = 0
            thetapower[i] = 0
            alphapower[i] = 0
            betapower[i] = 0
        else:
            # 计算 Delta 波频带内的平均功率
            idx_band = np.logical_and(freqs >= 1, freqs <= 4)
            deltapower[i] = np.mean(psd[idx_band])
            # 计算 Theta 波频带内的平均功率
            idx_band = np.logical_and(freqs >= 4, freqs <= 8)
            thetapower[i] = np.mean(psd[idx_band])
            # 计算 Alpha 波频带内的平均功率
            idx_band = np.logical_and(freqs >= 8, freqs <= 12)
            alphapower[i] = np.mean(psd[idx_band])
            # 计算 Beta 波频带内的平均功率
            idx_band = np.logical_and(freqs >= 12, freqs <= 30)
            betapower[i] = np.mean(psd[idx_band])

            # 求 Theta 和 Alpha 波频带内功率平均值的总体平均值
            mean_power[i] = (thetapower[i] + alphapower[i]) / 2
            # mean_power[i] = betapower[i]
            # mean_power[i] =  (betapower[i] + thetapower[i]) / 2
            # mean_power[i] = deltapower[i]
            # mean_power[i] =  (deltapower[i] + alphapower[i]) / 2



    return mean_power

def wavelet_transform(data):
    # 选择小波变换的类型和层数
    wavelet = 'db1'  # 小波类型，可以根据需要选择
    level = 1  # 小波变换的层数
    # 初始化一个数组，用于存储变换后的系数
    coeffs = np.empty_like(data)
    # 对每个序列进行小波变换
    for i in range(data.shape[0]):
        sequence_data = data[i, :]
        sequence_coeffs = pywt.wavedec(sequence_data, wavelet, level=level)
        coeffs[i, :] = np.concatenate([c[:len(sequence_data)] for c in sequence_coeffs])
    return coeffs