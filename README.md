# TripletChain
TripletLoss criterion for Chainer

## Usage example

```python
from chainer import Chain, Variable
from loss import triplet_loss

class TripletNet(Chain):

    def forward_once(self, x_data, train=True):
        x = Variable(x_data)
        h = self.layer1(x)
        ...
        y = self.layerN(h)
  
        return y
  
    def forward(self, a, p, n, train=True):
        y_a = self.forward_once(a, train)
        y_p = self.forward_once(p, train)
        y_n = self.forward_once(n, train)
  
        return triplet_loss(y_a, y_p, y_n)
```

## Sampling function example (using chainer)

```python
from scipy.spatial.distance import cdist
import numpy as np
import itertools

def sample_softnegative_triplets(X, y, xp, model, margin = 0.2):
    '''
    Returns soft-negative triplets as described in Google 2015 paper:
        http://arxiv.org/pdf/1503.03832.pdf
    X - train data
    y - labels
    xp - chainer.cupy/numpy container (depends wether you have Cuda or not,
    you can use simple NumPy if you dont have cuda)
    model - chainer model
    '''
    # forward passing X batch through network to get embeddings for every image
    embeddings = model.forward_once(X, train=False)
    try:
        emb = xp.asnumpy(embeddings.data)
    except Exception:
        emb = embeddings.data
    pairwise_distances = cdist(emb, emb, 'sqeuclidean')
    a_p = []
    triplets = []
    for label in np.unique(y):
        # forming a-p pairs or each class
        a_p.extend(list(itertools.combinations(np.where(y==label)[0], 2)))
    for a_i, p_i in a_p:
        # extracting current pair distance
        a_p_dist = pairwise_distances[a_i, p_i]
        # extracting all indexes of negative samples
        negatives = np.where(y!=y[a_i])[0]
        # subtracting current pair distance from negative pairs (ie a-n)
        diff = pairwise_distances[a_i, negatives] - a_p_dist
        # extracting soft negatives indexes, ie triplets with a-n distance greater than a-p and within margin
        soft_negatives = np.where((diff>0.) & (diff<margin))[0]
        # extracting index of most hard among soft-negatives
        if len(soft_negatives) > 0:
            # finding min index of differences, pointing to soft negatives index, pointing to negative index
            n_i = negatives[soft_negatives[diff[soft_negatives].argmin()]]
            # adding triplet indexes to result list
            triplets.append(np.array((a_i, p_i, n_i)))
    # return soft-negative triplets and AP samples amount
    return np.array(triplets, dtype=np.int32), len(a_p)
    
# Can be used in training like this:
def train_batch(X, y, xp, optimizer, model):
    triplets, AP_pairs = sample_softnegative_triplets(xp.asarray(X, dtype=xp.float32), y, xp, model)
    if len(triplets) == 0:
        return 0., AP_pairs, 0.
    a = xp.asarray(X[triplets[:, 0]], dtype=xp.float32)
    p = xp.asarray(X[triplets[:, 1]], dtype=xp.float32)
    n = xp.asarray(X[triplets[:, 2]], dtype=xp.float32)
    optimizer.zero_grads()
    loss = model.forward(a, p, n)
    loss.backward()
    optimizer.update()
    batch_loss = float(loss.data) * len(triplets)
    return batch_loss, AP_pairs, len(triplets)
```
