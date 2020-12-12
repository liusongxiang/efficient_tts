#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Duration predictor related modules."""

import torch
import torch.nn.functional as F

from nntts.layers.layer_norm import LayerNorm


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.

    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, 
                 num_spks=None, spk_embed_dim=None, spk_embed_integration_type='add'):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.n_layers = n_layers

        self.spk_embed_dim = spk_embed_dim
        self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            assert num_spks is not None, "num_spks has to be set."   
            self.spk_embedding = torch.nn.Embedding(num_spks, self.spk_embed_dim)
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, n_chans) 
            else:
                self.projection = torch.nn.Linear(idim + self.spk_embed_dim, n_chans)

        self.conv = torch.nn.ModuleList()       
        for idx in range(n_layers):
            in_chans = n_chans # idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(
                torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                torch.nn.ReLU(),
                LayerNorm(n_chans, dim=1),
                torch.nn.Dropout(dropout_rate)
            )]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, spembs=None, is_inference=False, to_round=True):
        if self.spk_embed_dim is not None:
            spembs = self.spk_embedding(spembs)
            xs = self._integrate_with_spk_embed(xs, spembs)
         
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  # (B, Tmax)

        if is_inference:
            if to_round:
                # NOTE: calculate in linear domain
                xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()  # avoid negative value
            else:
                xs = torch.clamp(xs.exp() - self.offset, min=0)
        
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None, spembs=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, x_masks, spembs, False)

    def inference(self, xs, x_masks=None, spembs=None, to_round=True):
        """Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, x_masks, spembs, True, to_round=to_round)

    def _integrate_with_spk_embed(self, hs, spembs):
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim)

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
