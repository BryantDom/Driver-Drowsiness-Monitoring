import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import accuracy_score

def EEGtest(model, device, test_x, test_y, subjectnum):
    model.eval()
    test_loss = 0
    correct = 0
    predict_total = []
    label_total = []
    with torch.no_grad():
        data, target = test_x.to(device), test_y.to(device)
        output, predict = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        # batch_loss = loss(predict, target)
        predict_val = np.argmax(predict.cpu().data.numpy(), axis=1)
        target = target.cpu().data.numpy()
        predict_total = np.append(predict_total, predict_val)
        acc_11 = accuracy_score(target, predict_val)
        label_val = target
        label_total = np.append(label_total, label_val)

        val_TP = ((predict_total == 1) & (label_total == 1)).sum().item()
        val_TN = ((predict_total == 0) & (label_total == 0)).sum().item()
        val_FN = ((predict_total == 0) & (label_total == 1)).sum().item()
        val_FP = ((predict_total == 1) & (label_total == 0)).sum().item()

        # 计算特异度（Sensitivity）和召回率（Recall）
        val_spe = val_TN / (val_FP + val_TN + 0.000001)
        val_rec = val_TP / (val_TP + val_FN + 0.000001)

        # 计算精确度（Precision）
        val_pre = val_TP / (val_TP + val_FP + 0.000001)

        # 计算F1分数
        val_f1 = 2 * (val_pre * val_rec) / (val_pre + val_rec + 0.000001)
        print("第", subjectnum, "的准确率是:", acc_11,"pre是:", val_pre, "spe:", val_spe)
        rec_11 = val_rec
        f1_11 = val_f1
        spe_11 = val_spe
        pre_11 = val_pre
    return acc_11, rec_11, f1_11, spe_11, pre_11

def EEGtrain(args, model, device, train_loader, test_x, test_y, optimizer, subjectnum):
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    acc_11Subject = EEGtest(model, device, test_x, test_y, subjectnum)
    return acc_11Subject, acc_11Subject
