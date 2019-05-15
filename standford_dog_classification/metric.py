# -*- coding:utf-8 -*-

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    print("output ",output)
    print("target ",target)
    maxk = max(topk)
    print("maxk ",maxk)
    batch_size = target.size(0)
    print("batch_size ", batch_size)
    _, pred = output.topk(maxk, 1, True, True)
    print("pred ",pred)

    pred_numpy = pred.numpy()
    print(pred_numpy)

    # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
    # input:输入张量, k = topk的k,dim:排序的维，largest:返回最大还是最小，sorted:返回是否排序
    pred = pred.t()
    print("pred ", pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))    # view(1,-1)将target多行拼接成一行
    print("correct ", correct)
    res = []
    print("topk ",topk)
    for k in topk:
        print("k ",k)
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        print("correct_k ",correct_k)
        res.append(correct_k.mul_(100.0 / batch_size))
        print("res ",res)
    return res
