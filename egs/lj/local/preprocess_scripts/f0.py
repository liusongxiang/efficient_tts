import numpy as np
from .melodia_pitch import extract_merged_pitch, sptk_extract_f0, log_interp_f0, interp_f0


f0_stats = {
    'song_shibei': {'floor': 120,
                    'ceil': 800},
    'song_guiniang': {'floor': 100,
                      'ceil': 1000},
    'vctk': {'floor': 20,
             'ceil': 600},
}


def downsample_note_pitch(cf0_data, frame_shift_ms, default_frame_shift_ms):
    if frame_shift_ms != default_frame_shift_ms:
        ratio = frame_shift_ms / default_frame_shift_ms
        sel_ids = [int(idx) for idx in np.arange(0, cf0_data.shape[0], ratio)]
        cf0_data = cf0_data[sel_ids]
    return cf0_data


def compute_cont_f0(wave_path,
                   f0_floor,
                   f0_ceil,
                   extract_frame_shift_ms,
                   save_frame_shift_ms,
                   method,
                   use_reaper,
                   ):
    assert method in ['melodia', 'reaper', 'rapt'], f"Unknown method: {method}"
    if method == 'melodia':
        # print("Using melodia ...")
        pitch = extract_merged_pitch(
            wave_path,
            F0_FLOOR=f0_floor,
            F0_CEIL=f0_ceil,
            FRAME_SHIFT=extract_frame_shift_ms,
            use_reaper=use_reaper)
    else:
        use_reaper = method == 'reaper'
        # print("Using reaper ...")
        pitch = sptk_extract_f0(
            wave_path,
            F0_FLOOR=f0_floor,
            F0_CEIL=f0_ceil,
            FRAME_SHIFT=extract_frame_shift_ms,
            use_reaper=use_reaper)
    
    cont_f0 = interp_f0(pitch, kind='slinear')
    cont_f0_save = downsample_note_pitch(cont_f0, save_frame_shift_ms, extract_frame_shift_ms)
    return cont_f0_save


def compute_f0_from_wav(speaker_name,
                        wave_path,
                        f0_config):

    f0_floor = f0_stats[speaker_name]['floor']
    f0_ceil = f0_stats[speaker_name]['ceil']
    cont_f0 = compute_cont_f0(wave_path=wave_path,
                              f0_floor=f0_floor,
                              f0_ceil=f0_ceil,
                              extract_frame_shift_ms=f0_config.extract_FRAME_SHIFT_MS,
                              save_frame_shift_ms=f0_config.save_FRAME_SHIFT_MS,
                              method=f0_config.method,
                              use_reaper=f0_config.use_reaper)
    return cont_f0


