import librosa
import librosa.filters
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile


def load_wav(path, sample_rate):
    return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # librosa.output.write_wav(path, wav.astype(np.int16), sample_rate)
    wavfile.write(path, sample_rate, wav.astype(np.int16))


def preemphasis(x, pre_emphasis):
    if pre_emphasis < 1.0:
        return signal.lfilter([1, -pre_emphasis], [1], x)
    else:
        return x


def inv_preemphasis(x, pre_emphasis):
    return signal.lfilter([1], [1, -pre_emphasis], x)


# def spectrogram(y, pre_emphasis, ref_level_db):
    # D = _stft(preemphasis(y, pre_emphasis))
    # S = _amp_to_db(np.abs(D)) - ref_level_db
    # return _normalize(S)

# def inv_spectrogram(spectrogram, ref_level_db, power=1.5):
  # '''Converts spectrogram to waveform using librosa'''
  # S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
  # return inv_preemphasis(_griffin_lim(S ** power))          # Reconstruct phase

def spectrogram(wave, 
                sample_rate,
                num_freq,
                frame_length_ms=50,
                frame_shift_ms=10,
                pre_emphasis=0.97,
                min_level_db=-100,
                ref_level_db=20):
    '''
    Args:
    wave: wav signal
    '''
    n_fft, hop_length, win_length = _stft_parameters(
        sample_rate, num_freq, frame_length_ms, frame_shift_ms)
    # 1. Compute STFT
    stft_spec = _stft(preemphasis(wave, pre_emphasis), 
                      configs=(n_fft, hop_length, win_length))
    # 2. Compute log spectrogram
    stft_db = _amp_to_db(np.abs(stft_spec)) - ref_level_db
    # Normalize to [0, 1]
    stft_db_norm = _normalize(stft_db, min_level_db)
    return stft_db_norm

def melspectrogram(wave, 
                   sample_rate,
                   num_freq,
                   num_mels=80,
                   frame_length_ms=50,
                   frame_shift_ms=10,
                   pre_emphasis=0.97,
                   fmin=40,
                   min_level_db=-100,
                   ref_level_db=20):
    '''
    Args:
    wave: wav signal
    '''
    n_fft, hop_length, win_length = _stft_parameters(
        sample_rate, num_freq, frame_length_ms, frame_shift_ms)
    # 1. Compute STFT
    stft_spec = _stft(preemphasis(wave, pre_emphasis), 
                      configs=(n_fft, hop_length, win_length))
    # 2. Compute Mel spectrogram 
    mel_spec_amp = _linear_to_mel(np.abs(stft_spec), num_freq, sample_rate, num_mels, fmin)
    # 3. Compute log Mel spectrogram
    mel_spec_db = _amp_to_db(mel_spec_amp) - ref_level_db
    # 4. Normalize to [0, 1]
    mel_spec_db_norm = _normalize(mel_spec_db, min_level_db)
    return mel_spec_db_norm

def _stft_parameters(sample_rate, num_freq, frame_length_ms, frame_shift_ms):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length


def _stft(y, configs):
    # n_fft, hop_length, win_length = _stft_parameters()
    n_fft, hop_length, win_length = configs
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram, num_freq, sample_rate, num_mels, fmin):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(num_freq, sample_rate, num_mels, fmin)
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis(num_freq, sample_rate, num_mels, fmin):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S, min_level_db):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

