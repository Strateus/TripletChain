"""
Created on Thu Nov 24 12:50:06 2015

TripletChainer

@author: Igor A. Stankevich (loknar at list.ru)
created with help of Alfredo Canziani, Torch TripletEmbedding (https://github.com/Atcold/torch-TripletEmbedding) author

License: MIT
"""

import numpy as np

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class TripletLoss(function.Function):

    """Triplet loss function."""

    def __init__(self, margin = 0.2, use_cudnn=True):
        self.margin = float(margin)
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape == in_types[2].shape,
            in_types[0].ndim == 2,
            in_types[1].ndim == 2,
            in_types[2].ndim == 2
        )

    def forward(self, inputs):
        # numpy or cupy interface selection
        xp = cuda.get_array_module(*inputs)
        a, p, n = inputs
        assert a.shape[0] == p.shape[0] == n.shape[0]
        # extracting batch size
        N = a.shape[0]
        # calculating elementwise differences between a and p, a and n matrices
        a_p_diff = a-p
        a_n_diff = a-n
        # calculating AP * AP.T
        AP = xp.dot(a_p_diff, a_p_diff.T)
        # calculating AN * AN.T
        AN = xp.dot(a_n_diff, a_n_diff.T)
        # subtracting squared positive distance from squared negative distance
        APN_diff = AP - AN
        # extracting diagonal elements - loss vector of every sample in batch
        APN_diag = xp.diag(APN_diff)
        # adding margin
        APN_diag_plus_margin = APN_diag + self.margin
        # thresholding loss with 0
        self.Li = xp.maximum(0, APN_diag_plus_margin)
        # summing diagonal elements to calculate batch loss
        cumulative_loss = xp.sum(self.Li)
        # averaging batch loss
        _loss = cumulative_loss / N
        
        return xp.array(_loss, dtype=xp.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        a, p, n = inputs
        N = a.shape[0]
        # calculating n-p, p-a and a-n elementwise differences
        n_p_diff = n-p
        p_a_diff = p-a
        a_n_diff = a-n
        # calculating Loss derivatives over a, n, p
        dLda = n_p_diff * 2 / N
        dLdp = p_a_diff * 2 / N
        dLdn = a_n_diff * 2 / N
        # masking gradients with zeros for triplets where loss was zero
        dLda[self.Li==0.] = 0.
        dLdp[self.Li==0.] = 0.
        dLdn[self.Li==0.] = 0.
        # applying previous gradient (elementwise multiplication)
        ga = (dLda * gy[0]).astype(xp.float32)
        gp = (dLdp * gy[0]).astype(xp.float32)
        gn = (dLdn * gy[0]).astype(xp.float32)
        
        return ga, gp, gn
        
def test():
    batch = 3
    embeddingSize = 5
    a = np.random.random((batch, embeddingSize))
    print '=== a ==='
    print a
    p = np.random.random((batch, embeddingSize))
    print '=== p ==='
    print p
    n = np.random.random((batch, embeddingSize))
    print '=== n ==='
    print n
    tl = TripletLoss(0.2, use_cudnn=False)
    loss = tl.forward((a,p,n))
    print '====== loss ======='
    print loss
    gradInput = tl.backward((a, p, n), (1,))
    print '=== grad a ==='
    print gradInput[0]
    print '=== grad p ==='
    print gradInput[1]
    print '=== grad n ==='
    print gradInput[2]
    
    d = 1e-6
    jacobian = {}
    zz = np.eye(3)
    for k in xrange(3):
        jacobian[k] = np.zeros_like(a)
        z = zz[k]
        for i in xrange(a.shape[0]):
            for j in xrange(a.shape[1]):
                pert = np.zeros_like(a)
                pert[i][j] = d
                outA = tl.forward((a - pert*z[0], p - pert*z[1], n - pert*z[2]))[0]
                outB = tl.forward((a + pert*z[0], p + pert*z[1], n + pert*z[2]))[0]
                jacobian[k][i][j] = (outB - outA)/(2*d)
    print '====== jacobian 1 ======='
    print jacobian[0]
    print '====== jacobian 2 ======='
    print jacobian[1]
    print '====== jacobian 3 ======='
    print jacobian[2]

def triplet_loss(a, p, n, margin=0.2, use_cudnn=True):
    """Triplet loss.

    """
    return TripletLoss(margin, use_cudnn)(a, p, n)





