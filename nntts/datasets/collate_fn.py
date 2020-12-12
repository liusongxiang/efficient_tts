import torch
import numpy as np


"""Collate functions"""

class TTSCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """ Prapare batch of text, duration, mel, and spkid        
        
        Args:
            batch: (phnseq, duration, mel, spkid)
        """
        batch_size = len(batch)
        text = [x[0] for x in batch]
        durations = [x[1] for x in batch]
        mels = [x[2] for x in batch]
        spkids = [x[3] for x in batch]
        
        mel_lengths = [x.shape[0] for x in mels]
        max_target_len = max(mel_lengths)
        max_index = mel_lengths.index(max_target_len)
        text_lengths = [x.shape[0] for x in text]
        max_input_len = max(text_lengths)

        text_padded = torch.LongTensor(batch_size, max_input_len)
        text_padded.zero_()
        spkid_tensor = torch.LongTensor(batch_size)
        for i in range(batch_size):
            cur_text = text[i]
            text_padded[i,:text_lengths[i]] = cur_text
            spkid_tensor[i] = int(spkids[i])

        num_mels = mels[0].shape[1]
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            mel_lengths[max_index] = max_target_len

        mel_padded = torch.FloatTensor(batch_size, max_target_len, num_mels)
        mel_padded.zero_()
        duration_padded = torch.LongTensor(batch_size, max_input_len)
        duration_padded.zero_()
        for i in range(batch_size):
            cur_mel = mels[i]
            mel_len = cur_mel.size(0)
            mel_padded[i,:mel_len,:] = cur_mel
            dur = durations[i]
            mel_len_padded = mel_lengths[i]
            if mel_len_padded > torch.sum(dur).item():            
                dur[-1] = dur[-1].item() + mel_len_padded - torch.sum(dur).item()
            duration_padded[i,:dur.size(0)] = dur

        return text_padded, torch.LongTensor(text_lengths), duration_padded, mel_padded, \
            torch.LongTensor(mel_lengths), spkid_tensor


class CommonDurationModelCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self,):
        pass

    def __call__(self, batch):
        """ Prapare batch of text, duration, mel, and spkid        
        
        Args:
            batch: (ndarray_1, ndarray_2, spkid), 
                e.g., ndarray_1 for ppg, ndarray_2 for durations/f0s
        """
        batch_size = len(batch)
        ppgs = [x[0] for x in batch]
        durations = [x[1] for x in batch]
        spkids = [x[2] for x in batch]
        
        lengths = [x.shape[0] for x in ppgs]
        max_length = max(lengths)
        
        # Obtain padded tensor batch
        ppg_dim = ppgs[0].shape[1]
        ppg_padded = torch.FloatTensor(batch_size, max_length, ppg_dim)
        ppg_padded.zero_()
        duration_padded = torch.LongTensor(batch_size, max_length)
        duration_padded.zero_()
        spkid_tensor = torch.LongTensor(batch_size)
        for i in range(batch_size):
            cur_ppg = ppgs[i]
            ppg_len = cur_ppg.size(0)
            ppg_padded[i,:ppg_len,:] = cur_ppg
            cur_dur = durations[i]
            duration_padded[i,:ppg_len] = cur_dur
            spkid_tensor[i] = int(spkids[i])

        return ppg_padded, torch.LongTensor(lengths), duration_padded, spkid_tensor

