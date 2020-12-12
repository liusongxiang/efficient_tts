#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Songxiang Liu
#  MIT License (https://opensource.org/licenses/MIT)

from typing import Dict
from typing import Tuple

import logging

import torch
import torch.nn.functional as F
import numpy as np

from nntts.losses.fastspeech_loss import FastSpeechLoss
from nntts.layers.duration_predictor import DurationPredictor
from nntts.utils.nets_utils import make_non_pad_mask, make_pad_mask, pad_list
from nntts.layers.initializer import initialize
from nntts.layers.efts_modules import ResConvBlock


class EfficientTTSCNN(torch.nn.Module):
    """EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture
    """
    def __init__(
        self,
        num_symbols: int, # 148
        odim: int = 80,
        symbol_embedding_dim: int = 512,
        n_channels: int = 512,
        n_text_encoder_layer: int = 5,
        n_mel_encoder_layer: int = 3,
        n_decoder_layer: int = 6,
        n_duration_layer: int = 2,
        k_size: int = 5,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        dropout_rate=0.1,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        duration_offset=1.0,
        sigma=0.01,
        sigma_e=0.5,
        delta_e_method_1=True,
        share_text_encoder_key_value=False,
        use_mel_query_fc=False,
    ):
        super().__init__()
        self.duration_offset = duration_offset
        self.sigma = sigma
        self.sigma_e = sigma_e
        self.delta_e_method_1 = delta_e_method_1  
        self.share_text_encoder_key_value = share_text_encoder_key_value

        self.text_embedding_table = torch.nn.Embedding(
            num_symbols, symbol_embedding_dim,
        )
        self.text_encoder = ResConvBlock(
            num_layers=n_text_encoder_layer,
            n_channels=n_channels,
            k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate,
            use_weight_norm=use_weight_norm,
        )
        self.text_encoder_key = torch.nn.Linear(
            n_channels, n_channels
        )
        if not share_text_encoder_key_value:
            self.text_encoder_value = torch.nn.Linear(
                n_channels, n_channels
            )
        self.mel_prenet = torch.nn.Sequential(
            torch.nn.Linear(odim, n_channels),
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            torch.nn.Dropout(dropout_rate),
        )
        self.mel_encoder = ResConvBlock(
            num_layers=n_mel_encoder_layer,
            n_channels=n_channels,
            k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate,
            use_weight_norm=use_weight_norm,
        )
        if use_mel_query_fc:
            self.mel_query_fc = torch.nn.Linear(
                n_channels, n_channels
            )
        else:
            self.mel_query_fc = None
        self.decoder = ResConvBlock(
            num_layers=n_decoder_layer,
            n_channels=n_channels,
            k_size=k_size,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            dropout_rate=dropout_rate,
            use_weight_norm=use_weight_norm,
        )
        self.mel_output_layer = torch.nn.Linear(n_channels, odim)
        # Duration predictor
        self.duration_predictor = DurationPredictor(
            idim=n_channels,
            n_layers=n_duration_layer,
            n_chans=n_channels,
            offset=duration_offset,
        )
         
        # define criterions
        self.criterion = FastSpeechLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward propagations.
        Args:
            text: Batch of padded text ids (B, T1).
            text_lengths: Batch of lengths of each input batch (B,).
            speech: Batch of mel-spectrograms (B, T2, num_mels)
            speech_lengths: Batch of mel-spectrogram lengths (B,)
        """
        device = text.device
        ## Prepare masks
        # [B, T1]
        text_mask = make_non_pad_mask(text_lengths).to(device)
        # [B, T2]
        mel_mask = make_non_pad_mask(speech_lengths).to(device)
        # [B, T1, T2]
        text_mel_mask = text_mask.unsqueeze(-1) & mel_mask.unsqueeze(1)
        
        # [B, T1, C] -> [B, C, T1]
        text_embedding = self.text_embedding_table(text).transpose(1, 2)

        ## text hidden
        # [B, C, T1] -> [B, T1, C]
        text_h = self.text_encoder(text_embedding).transpose(1, 2)
        text_key = self.text_encoder_key(text_h)
        if self.share_text_encoder_key_value:
            text_value = text_key
        else:
            text_value = self.text_encoder_value(text_h)

        _tmp_mask = ~(text_mask.unsqueeze(-1).repeat(1, 1, text_h.size(2)))
        text_key = text_key.masked_fill(_tmp_mask, 0.0)
        text_value = text_value.masked_fill(_tmp_mask, 0.0)

        ## mel hidden
        # [B, C, T2] -> [B, T2, C]
        mel_h = self.mel_prenet(speech).transpose(1, 2)
        mel_h = self.mel_encoder(mel_h).transpose(1, 2)
        if self.mel_query_fc is not None:
            mel_h = self.mel_query_fc(mel_h)

        ## Scaled dot-product attention
        alpha = self.scaled_dot_product_attention(mel_h, text_key, text_mask)
        alpha = alpha.masked_fill(~text_mel_mask, 0.0)
        
        ## Generate index mapping vector (IMV)
        text_index_vector = self.generate_index_vector(text_mask)
        # [B, T2]
        # print(text_lengths)
        imv = self.imv_generator(alpha, text_index_vector, mel_mask, text_lengths)
        
        ## Obtain alinged positions
        # [B, T1]
        e = self.get_aligned_positions(
            imv, text_index_vector, mel_mask, text_mask, sigma=self.sigma_e)
        e = e.squeeze(-1)
        
        ## Reconstructing alignment matrix
        # [B, T1, T2]
        reconst_alpha = self.reconstruct_align_from_aligned_position(
            e, delta=self.sigma, mel_mask=mel_mask, text_mask=text_mask
        ).masked_fill(~text_mel_mask, 0.0)

        ## Expanded text hidden to T2
        # [B, D, T2] 
        text_value_expanded = torch.bmm(
            text_value.transpose(1, 2), reconst_alpha
        )
        _tmp_mask_2 = ~(mel_mask.unsqueeze(1).repeat(1, text_value.size(2), 1))
        text_value_expanded = text_value_expanded.masked_fill(_tmp_mask_2, 0.0)
	
	## Decoder forward        
        mel_pred = self.decoder(text_value_expanded)
        mel_pred = self.mel_output_layer(mel_pred.transpose(1, 2))
        _tmp_mask_3 = ~(mel_mask.unsqueeze(-1).repeat(1, 1, 80))
        mel_pred = mel_pred.masked_fill(_tmp_mask_3, 0.0)

        # Prepare duration prediction target
        if self.delta_e_method_1:
            delta_e = torch.cat([e[:, :1], e[:,1:] - e[:, :-1]], dim=1).detach()
        else:
            B = speech_lengths.size(0)
            e = e.detach()
            e = torch.cat([e, torch.zeros(B, 1).to(e.device)], dim=1)
            for i in range(B):
                max_len = speech_lengths[i].cpu().item()
                max_index = text_lengths[i].cpu().item()
                e[i, max_index] = max_len
            delta_e = e[:, 1:] - e[:, :-1]

        log_delta_e = torch.log(delta_e + self.duration_offset)
        log_delta_e = log_delta_e.masked_fill(~text_mask, 0.0)
        
        ## DurationPredictor forward
        dur_pred = self.duration_predictor(text_value, ~text_mask)
        mel_loss, dur_loss = self.criterion(
            None, mel_pred, dur_pred, speech, log_delta_e, text_lengths, speech_lengths)

        loss = mel_loss + dur_loss

        stats = dict(
            loss=loss.item(), mel_loss=mel_loss.item(), duration_loss=dur_loss.item()
        )
        return loss, stats, imv, reconst_alpha, mel_pred, speech

    def inference(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor = None,
    ):
        """Inference.
        Args:
            text: Batch of padded text ids (B, T1).
            text_lengths: Batch of lengths of each input batch (B,).
        """
        device = text.device
        ## Prepare masks
        # [B, T1]
        # text_mask = make_non_pad_mask(text_lengths).to(device)
        
        # [B, T1, C] -> [B, C, T1]
        text_embedding = self.text_embedding_table(text).transpose(1, 2)

        ## text hidden
        # [B, C, T1] -> [B, T1, C]
        text_h = self.text_encoder(text_embedding).transpose(1, 2)
        text_key = self.text_encoder_key(text_h)
        if self.share_text_encoder_key_value:
            text_value = text_key
        else:
            text_value = self.text_encoder_value(text_h)

        # [1, T]
        delta_e = self.duration_predictor.inference(text_value, to_round=False)
        if self.delta_e_method_1:
            e = torch.cumsum(delta_e, dim=1)
        else:
            e = torch.cumsum(
                torch.cat([torch.zeros(1,1).to(delta_e.device), delta_e], dim=1), 
                dim=1
            )
        # print(e.shape)

        ## Reconstructing alignment matrix
        # [B, T1, T2]
        reconst_alpha = self.reconstruct_align_from_aligned_position(
            e, delta=self.sigma, 
            mel_mask=None, text_mask=None,
            trim_e=(not self.delta_e_method_1)
        )#.masked_fill(~text_mel_mask, 0.0)

        ## Expanded text hidden to T2
        # [B, D, T2] 
        text_value_expanded = torch.bmm(
            text_value.transpose(1, 2), reconst_alpha
        )
	
	## Decoder forward        
        mel_pred = self.decoder(text_value_expanded)
        mel_pred = self.mel_output_layer(mel_pred.transpose(1, 2))
        return mel_pred, reconst_alpha

    def generate_index_vector(self, text_mask):
        """Create index vector of text sequence.
        Args:
            text_mask: mask of text-sequence.[B, T1]
        Returns:
            index vector of text sequence. [B, T1]
        """
        device = text_mask.device
        B, T1 = text_mask.size()
        p = torch.arange(0, T1).repeat(B, 1).float().to(device)
        return p * text_mask
    
    def imv_generator(self, alpha, p, mel_mask, text_length):
        """Compute index mapping index (IMV) from alignment matrix alpha.
        Implementation of HMA.
        Args:
            alpha: scaled dot attention [B, T1, T2]
            p: index vector, output of generate_index_vector. [B, T1]
            mel_mask: mask of mel-spectrogram [B, T2]
            text_length: lengths of input text-sequence [B]
        Returns:
            Index mapping vector (IMV) [B, T2]
        """
        B, T1, T2 = alpha.size()
        # [B, T2]
        imv_dummy = torch.bmm(alpha.transpose(1, 2), p.unsqueeze(-1)).squeeze(-1)
        # [B, T2 - 1]
        delta_imv = torch.relu(imv_dummy[:, 1:] - imv_dummy[:, :-1])
        # [B, T2 - 1] -> [B, T2]
        delta_imv = torch.cat([torch.zeros(B, 1).type_as(alpha), delta_imv], -1)
        imv = torch.cumsum(delta_imv, -1) * mel_mask.float()
        # Get last element of imv
        last_imv, _ = torch.max(imv, dim=-1)
        # Avoid zeros
        last_imv = torch.clamp(last_imv, min=1e-8)
        # Multiply imv by a positive scalar to enforce 0 < imv < T1 - 1
        imv = imv / last_imv.unsqueeze(1) * (text_length.float().unsqueeze(-1) - 1)
        return imv

    def get_aligned_positions(self, imv, p, mel_mask, text_mask, sigma=0.5):
        """Compute aligned positions from imv
        Args:
            imv: index mapping vector [B, T2].
            p: index vector, output of generate_index_vector [B, T1].
            mel_mask: mask of mel-sepctrogram [B, T2].
            text_mask: mask of text sequences [B, T1].
            sigma: a scalar, default 0.5
        Returns:
            Aligned positions [B, T1].
        """
        # [B, T1, T2]
        energies = -1 * ((imv.unsqueeze(1) - p.unsqueeze(-1))**2) * sigma
        energies = energies.masked_fill(
            ~(mel_mask.unsqueeze(1).repeat(1, energies.size(1), 1)),-float('inf'))
        beta = torch.softmax(energies, dim=2)
        q = torch.arange(0, mel_mask.size(-1)).unsqueeze(0).repeat(imv.size(0), 1).float().to(imv.device)
        # Generate index vector of target squence.
        q = q * mel_mask.float()
        return torch.bmm(beta, q.unsqueeze(-1)) * text_mask.unsqueeze(-1)

    def reconstruct_align_from_aligned_position(
            self, e, delta=0.1, mel_mask=None, text_mask=None, trim_e=False):
        """Reconstruct alignment matrix from aligned positions.
        Args:
            e: aligned positions [B, T1].
            delta: a scalar, default 0.1
            mel_mask: mask of mel-spectrogram [B, T2], None if inference and B==1.
            text_mask: mask of text-sequence, None if B==1.
        Returns:
            alignment matrix [B, T1, T2].
        """
        if mel_mask is None:
            # inference phase
            # max_length = torch.round(e[:,-1] + (e[:,-1] - e[:, -2])).squeeze().item()
            max_length = torch.round(e[:,-1]).squeeze().item() 
            if trim_e:
                e = e[:, :-1]
        else:
            max_length = mel_mask.size(-1)
        q = torch.arange(0, max_length).unsqueeze(0).repeat(e.size(0), 1).to(e.device).float()
        if mel_mask is not None:
            q = q * mel_mask.float()
        energies = -1 * delta * (q.unsqueeze(1) - e.unsqueeze(-1))**2
        if text_mask is not None:
            energies = energies.masked_fill(
                ~(text_mask.unsqueeze(-1).repeat(1, 1, max_length)),
                -float('inf')
            )
        return torch.softmax(energies, dim=1)

    def scaled_dot_product_attention(self, query, key, key_mask):
        """
        Args:
            query: mel-encoder output [B, T2, D]
            key: text-encoder output [B, T1, D]
            key_mask: mask of text-encoder output [B, T1]
        Returns:
            attention alignment matrix alpha [B, T1, T2].
        """
        D = key.size(-1)
        T1 = key.size(1)
        T2 = query.size(1)
        # [B, T2, T1]
        scores = torch.bmm(query, key.transpose(-2, -1)) / np.sqrt(float(D))
        # [B, T2, T1]
        key_mask = ~(key_mask.unsqueeze(1).repeat(1, T2, 1))
        scores = scores.masked_fill(key_mask, -float('inf'))
        # [B, T2, T1]
        alpha = torch.softmax(scores, dim=-1).masked_fill(
            key_mask, 0.0
        )
        return alpha.transpose(-2, -1)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                # m.weight.data.normal_(0.0, 0.02)
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)    


if __name__ == "__main__":
    num_symbols = 148
    model = EfficientTTSCNN(num_symbols)
    B=2
    T1_1, T1_2 = 60, 80
    T2_1, T2_2 = 450, 500
    text_1 = torch.arange(T1_1)
    text_2 = torch.arange(T1_2)
    mel_1 = torch.rand(T2_1, 80)
    mel_2 = torch.rand(T2_2, 80)
    text = pad_list([text_1, text_2], 0)
    mel = pad_list([mel_1, mel_2], 0)
    text_lengths = torch.LongTensor([T1_1, T1_2])
    mel_lengths = torch.LongTensor([T2_1, T2_2])
    D = 512
    loss, stats = model(text, text_lengths, mel, mel_lengths)
    print(loss)
    print(stats)

