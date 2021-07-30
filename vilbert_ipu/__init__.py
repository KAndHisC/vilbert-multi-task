
import poptorch
from torch import nn


class RecomputationCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        
        return poptorch.recomputationCheckpoint(self.layer(*args, **kwargs))
        # result = self.layer(*args, **kwargs)
        # if len(result)==1:
        #     return poptorch.recomputationCheckpoint(result)
        # else:
        #     return tuple(poptorch.recomputationCheckpoint(y) for y in result)
