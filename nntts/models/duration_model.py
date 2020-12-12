
"""Duration Model."""

from typing import Dict
from typing import Tuple

import torch
import torch.nn.functional as F

from nntts.losses.duration_loss import DurationMSELoss
from nntts.layers.duration_predictor import DurationPredictor
from nntts.utils.nets_utils import make_non_pad_mask, make_pad_mask
from nntts.layers.initializer import initialize


class DurationModel(torch.nn.Module):
    """Duration model."""
    def __init__(
        self,
        idim: int,
        odim: int = 1,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 256, # 384
        duration_predictor_kernel_size: int = 3,
        num_spks: int = None,
        spk_embed_dim: int = None,        # speaker embedding dimension
        spk_embed_integration_type: str = "add",
        duration_predictor_dropout_rate: float = 0.1,
        use_masking: bool = True, # False
        use_weighted_masking: bool = False,
    ):
        super().__init__()
        
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.spk_embed_dim = spk_embed_dim
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        # set padding idx
        padding_idx = 0

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=idim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
            num_spks=num_spks,
            spk_embed_dim=self.spk_embed_dim,
            spk_embed_integration_type=spk_embed_integration_type
        )

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.criterion = DurationMSELoss(reduction=reduction)
    
    def _forward(self, xs, ilens, spkids=None, is_inference=False):
        d_masks = make_pad_mask(ilens).to(xs.device) #(B, Tmax)
        
        if is_inference:
            d_outs = self.duration_predictor(xs, d_masks, spkids)
            # Avoid negative nvalue
            d_outs = torch.clamp(torch.round(d_outs.exp() - 1.0), min=0).long()
        else:
            # Training mode
            d_outs = self.duration_predictor(xs, d_masks, spkids)
        return d_outs

    def forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        durations: torch.Tensor,
        spkids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = xs.size(0)
        
        # Forward propagation
        d_outs = self._forward(
            xs, ilens, spkids, is_inference=False
        )
        
        # Calculate loss
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(d_outs.device)
            d_outs = d_outs.masked_select(duration_masks)
            d_trgs = durations.masked_select(duration_masks)
        duration_loss = self.criterion(d_outs, d_trgs)

        stats = dict(
            loss=duration_loss.item()
        )
        return duration_loss, stats
    
    def inference(
        self,
        xs: torch.Tensor,
        spkids: torch.Tensor = None,
    ) -> torch.Tensor:
        # setup batch axis
        ilens = torch.tensor([xs.shape[1]], dtype=torch.long, device=xs.device)

        # inference
        d_outs = self._forward(xs, ilens, spkids=spkids, is_inference=True)
        return d_outs[0]
