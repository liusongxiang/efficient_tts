import torch
import torch.nn.functional as F
from nntts.losses.duration_loss import DurationMSELoss
from nntts.utils.nets_utils import make_non_pad_mask


class FastSpeechLoss(torch.nn.Module):
    """Loss function for Fastspeech model."""

    def __init__(
        self,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        use_mse: bool = True,
    ):
        """Initialize Fastspeech loss module.
        
        Args:
            use_masking (bool): Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss calculation.

        """
        super().__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        if use_mse:
            self.mel_criterion = torch.nn.MSELoss(reduction=reduction)
        else:
            self.mel_criterion = torch.nn.L1Loss(reduction=reduction)
        self.duration_criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, after_outs, before_outs, d_outs, ys, ds, ilens, olens):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = after_outs.masked_select(out_masks) if after_outs is not None else None
            ys = ys.masked_select(out_masks)

        # calculate loss
        mel_loss = self.mel_criterion(before_outs, ys)
        if after_outs is not None:
            mel_loss += self.mel_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)

        return mel_loss, duration_loss
