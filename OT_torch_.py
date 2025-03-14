import numpy as np
import torch
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances
from torch.autograd import Variable
import pdb


# 直接用于utils中OT函数
def cost_matrix_batch_torch(x, y):
    "Returns the cosine distance batchwise"
    # x is the image feature: bs * d * m * m
    # y is the audio feature: bs * d * nF
    # return: bs * n * m
    # print(x.size())
    bs = list(x.size())[0]
    D = x.size(1)
    assert (x.size(1) == y.size(1))
    # x = x.contiguous().view(bs, D, -1) # bs * d * m^2
    x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
    y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
    cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
    cos_dis = 1 - cos_dis  # to minimize this value
    # cos_dis = - cos_dis
    return cos_dis.transpose(2, 1)


# 用于下文GW_distance函数
def cos_batch_torch(x, y):
    "Returns the cosine distance batchwise"
    # x is the image feature: bs * d * m * m
    # y is the audio feature: bs * d * nF
    # return: bs * n * m
    # print(x.size())
    bs = x.size(0)
    D = x.size(1)
    assert (x.size(1) == y.size(1))
    x = x.contiguous().view(bs, D, -1)  # bs * d * m^2
    x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
    y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
    cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
    cos_dis = 1 - cos_dis  # to minimize this value
    # return cos_dis.transpose(2,1)
    # TODO:
    beta = 0.1
    min_score = cos_dis.min()
    max_score = cos_dis.max()
    threshold = min_score + beta * (max_score - min_score)
    res = cos_dis - threshold
    # res = torch.nn.ReLU()

    return torch.nn.functional.relu(res.transpose(2, 1))


# 直接用于utils中OT函数
def IPOT_distance_torch_batch_uniform(C, bs, n, m, iteration=50):
    C = C.float().cuda()
    T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration)
    temp = torch.bmm(torch.transpose(C, 1, 2), T)
    distance = batch_trace(temp, m, bs)
    return -distance


# 用于下文GW_torch_batch
def IPOT_torch_batch_uniform(C, bs, n, m, beta=0.5, iteration=50):
    # C is the distance matrix
    # c: bs by n by m
    sigma = torch.ones(bs, int(m), 1).cuda() / float(m)
    T = torch.ones(bs, n, m).cuda()
    A = torch.exp(-C / beta).float().cuda()
    for t in range(1):
        Q = A * T  # bs * n * m
        del T
        for k in range(iteration):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q, 1, 2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2, 1)
        del Q

    return T  # .detach()


# 用于下文GW_distance_uniform函数
def GW_distance(X, Y, p, q, sub=True, lamda=0.5, iteration=5, OT_iteration=20, **kwargs):
    """
	:param X, Y: Source and target embeddings , batchsize by embed_dim by n
	:param p, q: probability vectors
	:param lamda: regularization
	:return: GW distance
	"""

    Cs = cos_batch_torch(X, X).float().cuda()
    Ct = cos_batch_torch(Y, Y).float().cuda()
    if sub:
        Cs = kwargs['w_ss'] * Cs
        Ct = kwargs['w_tt'] * Ct
    # pdb.set_trace()
    bs = Cs.size(0)
    m = Ct.size(2)
    n = Cs.size(2)
    T, Cst = GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
    temp = torch.bmm(torch.transpose(Cst, 1, 2), T)
    distance = batch_trace(temp, m, bs)
    return distance


# 用于上文GW_distance函数
def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
    one_m = torch.ones(bs, m, 1).float().cuda()
    one_n = torch.ones(bs, n, 1).float().cuda()

    Cst = torch.bmm(torch.bmm(Cs ** 2, p), torch.transpose(one_m, 1, 2)) + \
          torch.bmm(one_n, torch.bmm(torch.transpose(q, 1, 2), torch.transpose(Ct ** 2, 1, 2)))  # bs by n by m
    gamma = torch.bmm(p, q.transpose(2, 1))  # outer product, init
    # gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
    for i in range(iteration):
        C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        # # Sinkhorn iteration
        # b = torch.ones(bs, m, 1).cuda()
        # K = torch.exp(-C_gamma/beta)
        # for i in range(50):cd
        # 	a = p/(torch.bmm(K, b))
        # 	b = q/torch.bmm(K.transpose(1,2), a)
        # gamma = a * K * b
        gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
    Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
    return gamma.detach(), Cgamma


# 直接用于utils中OT函数
def GW_distance_uniform(X, Y, sub, lamda=1e-1, iteration=5, OT_iteration=20, **kwargs):
    m = X.size(2)
    n = Y.size(2)
    bs = X.size(0)
    p = (torch.ones(bs, m, 1) / m).cuda()
    q = (torch.ones(bs, n, 1) / n).cuda()
    return GW_distance(X, Y, p, q, sub, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration, **kwargs)


# 用于上文GW_distance函数
def batch_trace(input_matrix, n, bs):
    a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
    b = a * input_matrix
    return torch.sum(torch.sum(b, -1), -1).unsqueeze(1)
