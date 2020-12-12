import os, sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from util import audio
from util.audio import load_wav, spectrogram, melspectrogram
from util import extract_alignment
from text import parse_pronounce
from collections import OrderedDict
import itertools
from f0 import compute_f0_from_wav


_start_buffer = 0.05
_end_buffer = 0.05


def load_fids(fids_txt):
    with open(fids_txt, 'rt', encoding='utf-8') as F:
        fids = [l.strip().split('|')[0].split('/')[1] for l in F.readlines() if l.strip()]
    return fids


def process_ppp(fid_list, 
               pos_prosody_phn_dir, 
               save_dir, 
               py_type="PHN_TONE", 
               use_seg=True, 
               use_head=True, 
               use_tail=False,
               ):
    """ Process pos-prosody-phn files to get text phn seq.
    Args:
        fid_list: [list] fids.
        pos_prosody_phn_dir: pos-prosody-phn directory.
        save_dir: feature directory to save text, phn, mel, spec etc.
        py_type: pinyin type. e.g., PHN_TONE.
        use_seg:
        use_head:
        use_tail:
    Returns:
        text_dict: [dict] fid to text phn-seq.
        main_fids: [list] fids.
    """
    text_dict = OrderedDict()

    def _extract_all(label_dir, fids):
        for fid in fids:
            text_fp = os.path.join(label_dir, fid + '.txt')
            if not os.path.exists(text_fp):
                warnings.warn('File not exists {}'.format(text_fp))
                continue
            with open(text_fp, 'rt', encoding='utf-8') as F:
                pronoun_lines = [l.strip() for l in F.readlines()]
            # key: list[phn]
            text_dict[fid] = parse_pronounce.parse_sent(
                pronoun_lines, py_type, use_head, use_tail)
            text_dict[fid] = text_dict[fid] + ["_JH_E", ]
    # Read phn-seq from ppp files
    _extract_all(pos_prosody_phn_dir, fid_list)

    train_txt = os.path.join(save_dir, 'train-{}.txt'.format(py_type))
    with open(train_txt, 'wt', encoding='utf-8') as F:
        for fid in text_dict.keys():
            if not use_seg:
                seg_list = ['_WORD_SEG#1', '_WORD_SEG#2', '_WORD_SEG#3', '_WORD_SEG#4', '_SPS_SEG']
                phn_seq_wseg = text_dict[fid]
                phn_seq_wseg_noseg = [phn for phn in phn_seq_wseg if phn not in seg_list]
                text_dict[fid] = phn_seq_wseg_noseg
            print("{}|{}".format(fid, " ".join(text_dict[fid])), file=F)
    return text_dict, fid_list


def build_frame_phn_idx_dict(text_dict, main_label_dir, main_fids, frame_shift_ms=10):
    """ Get frame-level text-phn position indices. 
    """
    frame_phn_idx_dict = OrderedDict()

    def _extract_all(label_dir, fids):
        for fid in fids:
            if fid not in text_dict:
                warnings.warn('pos-prosody-phn not exists {}'.format(fid))
                continue
            label_fp = os.path.join(label_dir, fid + '.lab')
            if not os.path.exists(label_fp):
                warnings.warn('{} frame_phn_idx is not extracted for label missing.'
                              .format(fid))
                continue
            with open(label_fp, 'rt', encoding='utf-8') as F:
                label_lines = [l.strip() for l in F.readlines()]
                # print(label_lines)
            frame_phn_idx, phn_seq_wseg_updated = extract_alignment.extract_frame_phn_idx(
                label_lines, text_dict[fid], frame_shift_ms, fid)
            # print(frame_phn_idx, phn_seq_wseg_updated)
            # sys.exit()
            text_dict[fid] = phn_seq_wseg_updated
            if not frame_phn_idx:
                warnings.warn('{} frame_phn_idx is not extracted for label mismatch.'.format(fid))
            else:
                frame_phn_idx_dict[fid] = frame_phn_idx
    _extract_all(main_label_dir, main_fids)

    return frame_phn_idx_dict, text_dict


def build_from_path(fid_list, 
                    base_dir,
                    wav_dir,
                    full_label_dir,
                    pos_prosody_phn_dir,
                    out_dir, 
                    num_workers, 
                    phn2idx_map,
                    audio_config,
                    f0_config,
                    tqdm=lambda x: x, 
                    trim_silence=False, 
                    expand_phn2idx=False, 
                    use_seg=True,
                    use_head=True,
                    use_tail=False,
                    py_type="PHN_TONE"):
    '''Preprocesses the speech dataset from a given input path into a given output directory.

    Args:
        fid_list: fidlist to process
        base_dir: the home directory of the whole dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        phn2idx_map: 
        tqdm: You can optionally pass tqdm to get a nice progress bar
    Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''
    # phn_seq_wseg: (dict) fid -> list[phn]
    phn_seq_wseg, main_fids = process_ppp(fid_list=fid_list, 
                                          pos_prosody_phn_dir=pos_prosody_phn_dir, 
                                          save_dir=out_dir, 
                                          py_type=py_type, 
                                          use_seg=use_seg, 
                                          use_head=use_head, 
                                          use_tail=use_tail,
                                          )
    valid_ids = list(phn_seq_wseg.keys())
    # print("Valid ids: ", len(valid_ids))
    # process full label
    frame_phn_idx_dict, phn_seq_wseg = build_frame_phn_idx_dict(phn_seq_wseg, full_label_dir, main_fids,
                                                                audio_config.frame_shift_ms)
    valid_ids = list(set(valid_ids).intersection(frame_phn_idx_dict.keys()))
    print("Valid ids: ", len(valid_ids))

    if expand_phn2idx:
        idx = 1
        for fid in valid_ids:
            for phn in phn_seq_wseg[fid]:
                if phn not in phn2idx_map:
                    while idx in phn2idx_map.values():
                        idx += 1
                    phn2idx_map[phn] = idx
                    idx += 1
    else:
        valid_ids_no_oov = []
        for fid in valid_ids:
            has_oov = False
            for i, phn in enumerate(phn_seq_wseg[fid]):
                if phn not in phn2idx_map:
                    print(phn)
                    has_oov = True
            if not has_oov:
                valid_ids_no_oov.append(fid)
            if has_oov:
                print(fid)
        valid_ids = valid_ids_no_oov         
    
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    for fid in valid_ids:
        # wave_path = os.path.join(wav_dir, fid + '.wav')
        wave_path = os.path.join(wav_dir, fid + '.wav')
        label_path = os.path.join(full_label_dir, fid + '.lab')
        futures.append(executor.submit(partial(_process_utterance, out_dir, fid, 
                    wave_path, label_path, phn_seq_wseg[fid], phn2idx_map, frame_phn_idx_dict[fid], 
                    trim_silence, audio_config, f0_config)))
        #futures.append(_process_utterance(out_dir, fid, wave_path, label_path, phn_seq_wseg[fid], phn2idx_map, frame_phn_idx_dict[fid], corpus_name, fid, lpcnet_feats_path, vocoder_type))
        index += 1
    return [future.result() for future in tqdm(futures)], phn2idx_map


def _process_utterance(out_dir, fid, wav_path, label_path, phn_seq, phn2idx_map, frame_phn_idx, 
                       trim_silence, audio_config, f0_config):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
        out_dir: The directory to write the spectrograms into
        fid: utterance id.
        wav_path: Path to the audio file containing the speech input
        label_path: Path to full label file.
        phn_seq: text-phn-seq from ppp file.
        phn2idx_map: map phn (str) to index (int).
        frame_phn_idx: frame-level text-phn positions.
        uttid: utterance id.
        trim_silence:
        audio_config:
        f0_config:
    Returns:
        tuple (expanded phn-seq length, fid)
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path, audio_config.sample_rate)

    # Trim Slience
    if trim_silence:
        start_offset, end_offset = _parse_labels(label_path)
        start_offset_frames = int(start_offset/(audio_config.frame_shift_ms/1000.0))
        end_offset_frames = int(end_offset/(audio_config.frame_shift_ms/1000.0))
    else:
        start_offset_frames = 0
        end_offset_frames = 1000000

    # Compute a mel-scale spectrogram for wavenet, wavernn 
    mel_spectrogram = melspectrogram(wave=wav,
                                     sample_rate=audio_config.sample_rate,
                                     num_freq=audio_config.num_freq,
                                     num_mels=audio_config.num_mels,
                                     frame_length_ms=audio_config.frame_length_ms,
                                     frame_shift_ms=audio_config.frame_shift_ms,
                                     pre_emphasis=audio_config.pre_emphasis,
                                     fmin=audio_config.fmin,
                                     min_level_db=audio_config.min_level_db,
                                     ref_level_db=audio_config.ref_level_db
                                     ).astype(np.float32)
    mel_spectrogram = mel_spectrogram.T  # -> [T, num_mels]
    stft = spectrogram(wave=wav,
                       sample_rate=audio_config.sample_rate,
                       num_freq=audio_config.num_freq,
                       frame_length_ms=audio_config.frame_length_ms,
                       frame_shift_ms=audio_config.frame_shift_ms,
                       pre_emphasis=audio_config.pre_emphasis,
                       min_level_db=audio_config.min_level_db,
                       ref_level_db=audio_config.ref_level_db
                       ).astype(np.float32)
    stft = stft.T  # -> [T, num_freq]
    cont_f0 = compute_f0_from_wav(speaker_name='song_guiniang',
                                  wave_path=wav_path,
                                  f0_config=f0_config)
    
    if len(cont_f0) > len(mel_spectrogram):
        cont_f0 = cont_f0[:len(mel_spectrogram)]
    elif len(cont_f0) < len(mel_spectrogram):
        pad_len = len(mel_spectrogram) - len(cont_f0)
        cont_f0 = np.pad(cont_f0, (0, pad_len), mode="constant", constant_values=cont_f0[-1])
    else:
        pass
    

    # Process phn sequence
    phn_seq_idx = np.asarray([phn2idx_map[i] for i in phn_seq])
    dur = np.asarray([len(list(group)) for (key,group) in itertools.groupby(frame_phn_idx)])
    pos_list = [key for (key, group) in itertools.groupby(frame_phn_idx)]
    txt_phnseq = np.array(phn_seq)[pos_list]
    
    # Cut off tailing silence
    cut_len = mel_spectrogram.shape[0] - sum(dur)
    if cut_len > 0:
        mel_spectrogram = mel_spectrogram[:-cut_len]
        stft = stft[:-cut_len]
        cont_f0 = cont_f0[:-cut_len]
    else:
        assert dur[-1] > -cut_len, "Not enough tailing silence."
        dur[-1] = dur[-1] + cut_len
    
    # Save features
    with open(f"{out_dir}/{fid}.phntone.txt", 'w') as f:
        f.write(' '.join(txt_phnseq) + '\n')
    np.save(f'{out_dir}/{fid}.mel.npy', mel_spectrogram, allow_pickle=False)
    np.save(f'{out_dir}/{fid}.stft.npy', stft, allow_pickle=False)
    np.save(f'{out_dir}/{fid}.dur.npy', dur, allow_pickle=False)
    np.save(f'{out_dir}/{fid}.f0.npy', cont_f0, allow_pickle=False)
    return len(frame_phn_idx), fid


def _parse_labels(path):
    labels = []
    if path is None:
        return (0, None)
    with open(os.path.join(path)) as f:
        for line in f:
            parts = line.strip().split(' ')
            labels.append((float(parts[0])/10000000.0, float(parts[1])/10000000.0))
    start = max(labels[0][1] - _start_buffer, 0)
    # Since there're _JH and _TAIL phones in the end of text,
    # and _JH also exists in the middle of some sentences,
    # I guess maybe we don't need to trim tail silence,
    # or we can set bigger end_buffer
    end = labels[-1][0] + _end_buffer
    return (start, end)
