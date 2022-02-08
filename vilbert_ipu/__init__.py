
import poptorch
from torch import nn


class RecomputationCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        
        return poptorch.recomputationCheckpoint(self.layer(*args, **kwargs))
