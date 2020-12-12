import re
import os, sys
import warnings
import math
import copy
import numpy as np

from functools import lru_cache, partial
from scipy.stats import norm
from text.parse_pronounce import special_phone_map


def extract_phn_timestamp(label_lines):
    """ Extract phn boundaries (in second) from full label lines.
    Returns:
        A list of [phn, start_time, end_time]
    """
    phn_timestamp = []
    for line in label_lines:
        s, e, full_label = line.split()
        phn = re.match('-(\w+)+', full_label).group(1)
        tone = re.search('&(\d)+', full_label).group(1)
        reverse_idx = re.search('~(\d)#', full_label).group(1)
        stress = re.search('\+(\d)!', full_label).group(1)
        if reverse_idx == '1':
            phn = phn + tone
        elif phn.isupper():
            phn = phn + stress

        try:
            phn = special_phone_map[phn]
        except KeyError:
            pass

        phn_timestamp.append([phn, int(s) / 1e7, int(e) / 1e7])
    return phn_timestamp


def _rm_digits(s):
    return ''.join([c for c in s if not c.isdigit()])

def strip_digit(s):
  head = s.rstrip('0123456')
  return head

def get_phone_idx_map(label_phns, text_phns, fid):
    """ Get position map from label-phn-seq to text-phn-seq.
    Rules:
        1. The pure phn-seqs should have the same length.
        2. Ignore _WORD_SEG#* in text-phn-seq.
        3. Ignore _SPS_SEG in text-phn-seq.
        4. Map sil/pau in label-phn-seq to pronunciation/word-seg in text-phn-seq.
        5. TODO
        6. Mapping the same base phns.
    """
    pure_label_phns = [p for p in label_phns if p != 'pau' and p != 'sil']
    pure_text_phns = [p for p in text_phns if not (p.startswith('_') or p.isdigit())]
    if len(pure_label_phns) != len(pure_text_phns):
        warnings.warn('number of phonemes mismatch {} vs {}'
                      .format(len(pure_label_phns), len(pure_text_phns)))
        print(fid)
        print('phnSeq from full-label:', ' '.join(pure_label_phns))
        print('phnSeq from ppp:       ', ' '.join(pure_text_phns))
        sys.exit()
        return None, None

    lp_idx = 0
    tp_idx = 0
    lp_tp_idx_map = dict()
    try:
        while lp_idx < len(label_phns):
            if '_WORD_SEG' in text_phns[tp_idx]:
                tp_idx += 1
                continue
            if text_phns[tp_idx].isdigit():
                tp_idx += 1
                continue
            if text_phns[tp_idx] == '_SPS_SEG':
                tp_idx += 1
                continue
            if label_phns[lp_idx] in ['sil', 'pau']:
                # punc and word seg startswith `_`
                if text_phns[tp_idx].startswith('_'):
                    lp_tp_idx_map[lp_idx] = tp_idx
                else:
                    if text_phns[tp_idx-2] == '_WORD_SEG#2':
                        text_phns.insert(tp_idx, 'pau#2')
                        lp_tp_idx_map[lp_idx] = tp_idx
                    elif text_phns[tp_idx-2] == '_WORD_SEG#3':
                        text_phns.insert(tp_idx, 'pau#3')
                        lp_tp_idx_map[lp_idx] = tp_idx
                    else:
                        raise ValueError('Full label pause must correspond to '
                                       'text label _HEAD, _WORD_SEG or _PUNC')
            else:
                if label_phns[lp_idx] == text_phns[tp_idx] or \
                        (strip_digit(label_phns[lp_idx])) == text_phns[tp_idx]:
                    lp_tp_idx_map[lp_idx] = tp_idx
                elif (label_phns[lp_idx].isupper()
                      and _rm_digits(label_phns[lp_idx]) == _rm_digits(text_phns[tp_idx])):
                    lp_tp_idx_map[lp_idx] = tp_idx
                elif label_phns[lp_idx].isupper() and '_' not in text_phns[tp_idx]:
                    lp_tp_idx_map[lp_idx] = tp_idx
                    warnings.warn('trigger confusion pair {} {}'
                                  .format(label_phns[lp_idx], text_phns[tp_idx]))
                else:
                    tp_idx += 1
                    continue
            lp_idx += 1
            tp_idx += 1
    except ValueError as e:
        warnings.warn('{}, but {} -> {}'.format(str(e), label_phns[lp_idx], text_phns[tp_idx]))
        print(fid)
        print(label_phns)
        print(text_phns)
        # sys.exit()
        return None, None
    except IndexError as e:
        warnings.warn('{}, phoneme mismatch'.format(str(e)))
        return None, None
    return lp_tp_idx_map, text_phns


def timestamp_to_frame_idx(timestamp_s, frame_shift_ms):
    return int(math.ceil(timestamp_s * 1000 / frame_shift_ms))


def get_frame_att_label(label_phn_timestamps, text_phns, frame_shift_ms, fid):
    """ Get frame-level text-phn position list.
    Args:
        label_phn_timestamps: phn-seq extracted from full-label lines and its
            corresponding start-time and end-time.
        text_phns: text phn-seq extracted from ppp file.
    Returns:
        frame_att_label: [list] frame-level text-phn postion.
        text_phns: updated text phn-seq (a list).
    """
    # to which input phn the output frame corresponds.
    label_phns = [phn[0] for phn in label_phn_timestamps]
    lp_tp_idx_map, text_phns = get_phone_idx_map(label_phns, text_phns, fid)
    if not lp_tp_idx_map:
        return None, None

    frame_att_label = []
    for lp_idx, (_, s, e) in enumerate(label_phn_timestamps):
        tp_idx = lp_tp_idx_map[lp_idx]
        start_frame = timestamp_to_frame_idx(s, frame_shift_ms)
        end_frame = timestamp_to_frame_idx(e, frame_shift_ms)
        frame_att_label.extend([tp_idx] * (end_frame - start_frame))
    # frame_att_label: [0, 0, 3, 3, 3, ..., len(tp)-1, len(tp)-1, ...] 
    # text_phns: insert pau#2 and pau#3 if applicable.
    return frame_att_label, text_phns


def remove_silence_from_timestamp(
        label_phn_timestamps, head_tail_sil_dur_ms, middle_sil_dur_ms,
        crop_all=True):
    ht_sil_dur_s = head_tail_sil_dur_ms / 1000.0
    m_sil_dur_s = middle_sil_dur_ms / 1000.0

    label_phn_timestamps = copy.deepcopy(label_phn_timestamps)

    def get_phn_dur(idx):
        return label_phn_timestamps[idx][2] - label_phn_timestamps[idx][1]

    def default_modify_phn_timestamp(idx, cp):
        label_phn_timestamps[idx][1] -= cp
        label_phn_timestamps[idx][2] -= cp

    num_phns = len(label_phn_timestamps)
    cropped_period = 0.
    for i in range(0, num_phns):
        sil_dur_s = (ht_sil_dur_s
                     if i == 0 or i == num_phns - 1 else m_sil_dur_s)
        if (label_phn_timestamps[i][0] in ('sil', 'pau')
                and get_phn_dur(i) > sil_dur_s):
            if not crop_all and 0 < i < num_phns - 1:
                default_modify_phn_timestamp(i, cropped_period)
            else:
                pau_cropped_period = get_phn_dur(i) - sil_dur_s
                label_phn_timestamps[i][1] -= cropped_period
                cropped_period += pau_cropped_period
                label_phn_timestamps[i][2] -= cropped_period
        else:
            default_modify_phn_timestamp(i, cropped_period)
    return label_phn_timestamps


@lru_cache(maxsize=1)
def get_normal_smooth_label(label, num_cate, scale=0.5, win_width=5):
    cdf = partial(norm.cdf, loc=label, scale=scale)
    if not win_width:
        truncated_factor = cdf(num_cate - 0.5) - cdf(-0.5)
        smooth_label = (cdf(np.arange(num_cate) + 0.5) - cdf(np.arange(num_cate) - 0.5)) / truncated_factor
    else:
        # assert win_width % 2 == 1 and win_width <= num_cate
        left_width = right_width = win_width // 2
        trunc_left = label - left_width if label - left_width >= 0 else 0
        # truncated_right is not included in the window
        trunc_right = (label + right_width + 1
                       if label + right_width + 1 <= num_cate else num_cate)
        truncated_scale = cdf(trunc_right - 0.5) - cdf(trunc_left - 0.5)
        smooth_label = np.zeros(shape=[num_cate])
        smooth_label[trunc_left: trunc_right] = (
                cdf(np.arange(trunc_left, trunc_right) + 0.5) -
                cdf(np.arange(trunc_left, trunc_right) - 0.5)) / truncated_scale
    return smooth_label


def get_sent_normal_smooth_label(att_label, scale=0.5, win_width=5):
    num_cate = att_label[-1] + 1
    sent_smooth_label = np.array(
        [get_normal_smooth_label(al, num_cate, scale, win_width)
         for al in att_label]
    )
    return sent_smooth_label.astype(np.float32)


def get_sent_one_hot_label(att_label):
    num_cate = att_label[-1] + 1
    sent_label = np.zeros([att_label.size, num_cate], dtype=np.float32)
    sent_label[np.arange(att_label.size), att_label.flatten()] = 1.0
    return sent_label.reshape(list(att_label.shape) + [num_cate])


def extract_frame_phn_idx(label_lines, text_phns, frame_shift_ms, fid):
    """ Extract frame-level phn indices.
    Args:
        label_lines: full label lines from one lab file.
        text_phns: corresponding text phn-seq (a list).
        frame_shift_ms: spectral feature frame shift.
    Returns:
        frame_att_label: frame-level text-phn position list. 
        text_phns: updated text phns, added pau#2/pau#3 when applicable.
    """
    label_phn_timestamps = extract_phn_timestamp(label_lines)
    frame_att_label, text_phns = get_frame_att_label(
        label_phn_timestamps, text_phns, frame_shift_ms, fid)
    return frame_att_label, text_phns
