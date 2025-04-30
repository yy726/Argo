import torch
import torch.nn as nn

"""
This is a trial to implement the DCN v2 module in the paper

    DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
    https://arxiv.org/pdf/2008.13535

with certain simplification

    1. to make it easier for understanding, the low rank approx of the W matrix is not implemented

Code reference:
    1. https://gist.github.com/lzqlzzq/4cd81048119dfe6baf2acec85d1fe790
    2. https://github.com/pytorch/torchrec/blob/main/torchrec/modules/crossnet.py
"""


class CrossNet(nn.Module):
    """
        CrossNet performs the feature crossing to capture the higher order feature
        interactions.
    """

    def __init__(self, num_layers: int = 3):
        super().__init__()
        # create the layers via the sequential
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128, bias=True),
                nn.BatchNorm1d(128),
            )
        ] for _ in range(num_layers))

    def forward(self, x0):
        """
            x0: B x D

            In the original feature, x0 is the output of concat of the dense vector
            after the embedding lookup. This support varlen format of the embedding
            dimension for each categorical features. In some impl (e.g. ref 1), the
            x0 is represented as B x seq x D, where the batch norm need some change
            to be integrated with a transpose layer
        """
        x = x0
        # This is fine but might not very efficient, Pytorch would generate temporary
        # activation of x for the auto differential graph computation to compute the
        # gradient correctly 
        for layer in self.layers:
            x = layer(x) * x0 + x
        return x


class DeepNet(nn.Module):
    """
        DeepNet is a regular MLP with non linear activations
    """
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(),
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DCNV2(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == "__main__":
    pass