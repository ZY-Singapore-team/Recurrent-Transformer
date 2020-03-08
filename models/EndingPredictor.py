from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

class SentNumModel(nn.module):
    def __init__(self, max_num):
        super(SentNumModel, self).__init__()
        
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)