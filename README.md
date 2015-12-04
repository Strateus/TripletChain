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
