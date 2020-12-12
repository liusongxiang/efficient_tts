"""Layer normalization module."""

import torch


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    
    Args:
        nout: output dim size
        dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super().__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        
        Args:
            x (torch.Tensor): input tensor
        Returns: 
            layer normalized tensor
        """
        if self.dim == -1:
            return super().forward(x)
        else:
            return super().forward(x.transpose(1, -1)).transpose(1, -1)

