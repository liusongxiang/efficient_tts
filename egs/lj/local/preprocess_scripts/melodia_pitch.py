import vamp
import resampy
import numpy as np
import io
import pyreaper
import pysptk
import sox
import tempfile
import soundfile
import os
import itertools

from scipy.interpolate import interp1d, CubicSpline
from scipy.io import wavfile


def sox_resample(wave_data, sr, target_sr):
    src_wav = os.path.join(
        '/tmp', next(tempfile._get_candidate_names()) + '.wav')
    trg_wav = os.path.join(
        '/tmp', next(tempfile._get_candidate_names()) + '.wav')
    soundfile.write(src_wav, wave_data, sr, 'PCM_16')
    tfm = sox.Transformer()
    tfm.rate(target_sr, 'v')
    tfm.build(src_wav, trg_wav)
    wave_data, _ = soundfile.read(trg_wav)
    os.remove(src_wav)
    os.remove(trg_wav)
    return wave_data


def melodia_extract_pitch(wave_fp, F0_FLOOR=120, F0_CEIL=800, FRAME_SHIFT=5):
    fs = 44100
    step_size = int(np.round(128. / 44100 * fs))
    block_size = int(np.round(2048. / 44100 * fs))

    # load audio using librosa
    # print("Loading audio...")
    data, sr = soundfile.read(wave_fp)

    # mixdown to mono if needed
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data.mean(axis=1)

    # resample to 44100 if needed
    if sr != fs:
        # data = resampy.resample(data, sr, fs)
        data = sox_resample(data, sr, fs)
        sr = fs

    # extract melody using melodia vamp plugin
    # print("Extracting melody f0 with MELODIA...")
    melody = vamp.collect(
        data, sr, "mtg-melodia:melodia",
        parameters={"voicing": 0.6,
                    "minfqr": F0_FLOOR,
                    "maxfqr": F0_CEIL},
        step_size=step_size, block_size=block_size)

    # hop = melody['vector'][0]
    pitch = melody['vector'][1]

    # impute missing 0's to compensate for starting timestamp
    pitch = np.insert(pitch, 0, [0]*8)

    uv_idx = pitch <= 0.
    pitch[uv_idx] = 0.

    pitch_frame_shift = step_size / fs
    assert FRAME_SHIFT > pitch_frame_shift
    ratio = pitch_frame_shift / (FRAME_SHIFT * 1e-3)
    target_len = int(np.ceil(pitch.shape[0] * ratio))
    sel_idx = [int(np.floor(i / ratio)) for i in range(target_len)]
    pitch = pitch[sel_idx]
    return pitch


def load_wave(wave_fp):
    sr, y = wavfile.read(wave_fp)
    if y.dtype != np.int16:
        buffer = io.BytesIO()
        soundfile.write(buffer, y, sr, 'PCM_16', format='raw')
        y = np.frombuffer(buffer.getvalue(), dtype=np.int16)
        buffer.close()
    return sr, y


def extract_f0(y, sampling_rate, f0_floor, f0_ceil, frame_shift_ms, use_reaper=True):
    if use_reaper:
        pm_times, pm, f0_times, f0, corr = pyreaper.reaper(
            y,
            fs=sampling_rate,
            minf0=f0_floor,
            maxf0=f0_ceil,
            frame_period=frame_shift_ms / 1000.0)
        f0[f0 == -1.] = 0.
    else:
        f0 = pysptk.sptk.rapt(
            y.astype(np.float32),
            fs=sampling_rate,
            min=f0_floor,
            max=f0_ceil,
            hopsize=int(frame_shift_ms / 1000.0 * sampling_rate),
            voice_bias=0.3)
    return f0.astype(np.float32)


def sptk_extract_f0(wave_fp, F0_FLOOR=120, F0_CEIL=800, FRAME_SHIFT=5, use_reaper=True):
    method = 'reaper' if use_reaper else 'rapt'
    # print("Extracting {} f0".format(method))
    sr, y = load_wave(wave_fp)
    try:
        f0 = extract_f0(
        y, sr, F0_FLOOR, F0_CEIL, FRAME_SHIFT, use_reaper=use_reaper)
    except:
        print('Unexcept')
    return f0


def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths


def merge_melodia_rapt(melodia_f0, rapt_f0, thres=20):
    min_len = min(melodia_f0.shape[0], rapt_f0.shape[0])
    melodia_f0 = melodia_f0[:min_len]
    rapt_f0 = rapt_f0[:min_len]

    uv_union = np.logical_or(rapt_f0 == 0., melodia_f0 == 0.)
    merged_lf0 = np.copy(melodia_f0)
    merged_lf0[uv_union] = 0.

    rapt_v_melodia_uv = np.logical_and(rapt_f0 != 0., merged_lf0 == 0.)
    run_vales, run_starts, run_lengths = find_runs(rapt_v_melodia_uv)
    rv_muv_sidx = run_starts[np.where(run_vales)[0]]
    rv_muv_len = run_lengths[np.where(run_vales)[0]]
    for i, l in zip(rv_muv_sidx, rv_muv_len):
        if l < thres:
            continue
        else:
            merged_lf0[i: i + l] = rapt_f0[i: i + l]
    return merged_lf0


def extract_merged_pitch(wave_fp, F0_FLOOR=120, F0_CEIL=800, FRAME_SHIFT=5, use_reaper=False):
    melodia_pitch = melodia_extract_pitch(
        wave_fp, F0_FLOOR=F0_FLOOR, F0_CEIL=F0_CEIL, FRAME_SHIFT=FRAME_SHIFT)
    # print('Got melodia pitch...')
    rapt_pitch = sptk_extract_f0(
        wave_fp, F0_FLOOR=F0_FLOOR, F0_CEIL=F0_CEIL, FRAME_SHIFT=FRAME_SHIFT,
        use_reaper=use_reaper)
    # print('Got rapt pitch...')
    merged_pitch = merge_melodia_rapt(melodia_pitch, rapt_pitch)
    # print('Got merged pitch...')
    return merged_pitch


def log_interp_f0_(f0, kind='cubic'):
    voiced_mask = f0 > 1.
    voiced_idx = np.where(voiced_mask)[0]
    if np.sum(voiced_mask) == 0:
        return None
    log_voiced_values = np.log(f0[voiced_mask])
    if kind == 'linear':
        log_interp_f0 = np.interp(np.arange(f0.size), voiced_idx, log_voiced_values)
    elif kind in ['cubic', 'slinear']:
        mean_log_f0 = np.mean(log_voiced_values)
        if voiced_idx[0] > 0:
            voiced_idx = np.insert(voiced_idx, 0, 0)
            log_voiced_values = np.insert(log_voiced_values, 0, mean_log_f0)
        if voiced_idx[-1] < f0.size - 1:
            voiced_idx = np.append(voiced_idx, f0.size - 1)
            log_voiced_values = np.append(log_voiced_values, mean_log_f0)
        if kind == 'cubic':
            interp_func = CubicSpline(voiced_idx, log_voiced_values, bc_type='natural')
        else:
            interp_func = interp1d(voiced_idx, log_voiced_values, kind='slinear')
        log_interp_f0 = interp_func(np.arange(f0.size))
    else:
        raise ValueError('Not supported F0 interpolation type {}'.format(kind))
    return log_interp_f0.astype(np.float32)


def log_interp_f0(f0, kind='cubic'):
    voiced_mask = f0 > 1.
    voiced_idx = np.where(voiced_mask)[0]
    if np.sum(voiced_mask) == 0:
        return None
    voiced_values = f0[voiced_mask]
    if kind == 'linear':
        interp_f0 = np.interp(np.arange(f0.size), voiced_idx, voiced_values)
    elif kind in ['cubic', 'slinear']:
        mean_log_f0 = np.mean(voiced_values)
        if voiced_idx[0] > 0:
            voiced_idx = np.insert(voiced_idx, 0, 0)
            voiced_values = np.insert(voiced_values, 0, mean_log_f0)
        if voiced_idx[-1] < f0.size - 1:
            voiced_idx = np.append(voiced_idx, f0.size - 1)
            voiced_values = np.append(voiced_values, mean_log_f0)
        if kind == 'cubic':
            interp_func = CubicSpline(voiced_idx, voiced_values, bc_type='natural')
        else:
            interp_func = interp1d(voiced_idx, voiced_values, kind='slinear')
        interp_f0 = interp_func(np.arange(f0.size))
    else:
        raise ValueError('Not supported F0 interpolation type {}'.format(kind))
    log_interp_f0 = np.log(interp_f0)
    return log_interp_f0.astype(np.float32)


def interp_f0(f0, kind='slinear'):
    # print('interp_f0...')
    voiced_mask = f0 > 1.
    voiced_idx = np.where(voiced_mask)[0]
    if np.sum(voiced_mask) == 0:
        print('no voiced segs...')
        return None
    voiced_values = f0[voiced_mask]
    if kind == 'linear':
        interp_f0 = np.interp(np.arange(f0.size), voiced_idx, voiced_values)
    elif kind in ['cubic', 'slinear']:
        mean_f0 = np.mean(voiced_values)
        if voiced_idx[0] > 0:
            voiced_idx = np.insert(voiced_idx, 0, 0)
            voiced_values = np.insert(voiced_values, 0, mean_f0)
        if voiced_idx[-1] < f0.size - 1:
            voiced_idx = np.append(voiced_idx, f0.size - 1)
            voiced_values = np.append(voiced_values, mean_f0)
        if kind == 'cubic':
            interp_func = CubicSpline(voiced_idx, voiced_values, bc_type='natural')
        else:
            interp_func = interp1d(voiced_idx, voiced_values, kind='slinear')
        interp_f0 = interp_func(np.arange(f0.size))
        # print('Get slinear f0...')
    else:
        raise ValueError('Not supported F0 interpolation type {}'.format(kind))
    return interp_f0.astype(np.float32)
