import os

from util import register


mandarin_initial_list = ["b", "ch", "c", "d", "f", "g", "h", "j", "k", "l",
                         "m", "n", "p", "q", "r", "sh", "s", "t", "x", "zh",
                         "z"]

# fuse rear case to avoid OOV
special_phone_map = {}

# punc list
punc_list = ['_FH', '_MH', '_DUN', '_DH', '_WH', '_OPUNC']
special_phn_list = ['_WORD_SEG#1', '_WORD_SEG#2', '_WORD_SEG#3', '_WORD_SEG#4', '_HEAD', '_SPS_SEG', '_JH_E', '_WH_E', '_TH_E']

# func puncs
punc_map = {
    '_FH': '_FH',
    '_MH': '_MH',
    '_DUN': '_DUN',
    '_DH': '_DH',
    '_WH': '_WH',
    '_TH': '_TH',
    '_DYH': '_OPUNC',
    '_KH': '_OPUNC',
    '_PZH': '_OPUNC',
    '_SLH': '_OPUNC',
    '_SMH': '_OPUNC',
    '_SYH': '_OPUNC',
    '_YD': '_OPUNC'}

final_punc_map = {
    '_DH_E': '_JH_E',
    '_JH': '_DH',
    '_OPUNC_E': '_JH_E'}

parse_pinyin_methods = {}
parse_line_methods = {}
parse_sent_methods = {}

def split_phone_tone(s):
  head = s.rstrip('0123456')
  if len(head) == len(s):
    phn_tone = [s]
  else:
    tail = s[len(head):]
    phn_tone = [head, tail]
  return phn_tone

@register.register('PHN_TONE_SEP', parse_pinyin_methods)
def parse_pinyin_phn_tone_sep(py):
    phns = py.split('-')
    phns_tone = []
    for i in phns:
      if i in special_phone_map:
        i = special_phone_map[i]
      phns_tone.extend(split_phone_tone(i))

    outputs = []
    if py.islower():
        outputs.extend(phns_tone)
    else:
        outputs.extend(phns_tone)
    return outputs

@register.register('PHN_TONE', parse_pinyin_methods)
def parse_pinyin_phn_tone(py):
    phns = py.split('-')
    outputs = []
    if py.islower():
        if len(phns) == 1:
            outputs.extend([phns[0]])
        else:
            yun_tail = phns[-1]
            if yun_tail in special_phone_map:
                yun_tail = special_phone_map[yun_tail]
            outputs.extend(phns[:-1] + [yun_tail])
    else:
        for phn in phns:
            if phn in special_phone_map:
                outputs.append(special_phone_map[phn])
            else:
                outputs.append(phn)
    return outputs


def parse_pinyin(pronoun_line, py_type):
    parts = pronoun_line.split()
    pinyin_str = parts[-1]
    pinyins = [py for py in pinyin_str.split("|")
               if py != ""]
    try:
        outputs = []
        for py in pinyins:
            outputs.extend(['_SPS_SEG'])
            outputs.extend(parse_pinyin_methods[py_type](py))
    except KeyError:
        raise ValueError('parse_pinyin for [{}] is not implemented'.format(py_type))
    return outputs


def parse_punct(pronoun_line):
    parts = pronoun_line.split()
    punct_part = parts[3]
    prosody_word_seg_sign = parts[-2]
    if prosody_word_seg_sign == '#0':
       suffix = []
    else:
       if punct_part != '0':
           punc = '_' + punct_part.upper()
           if punc in punc_map:
               punc = punc_map[punc]
           suffix = ['_WORD_SEG' + prosody_word_seg_sign] + [punc]
       else:
           suffix = ['_WORD_SEG' + prosody_word_seg_sign]
    return suffix


def parse_pos(pronoun_line):
    parts = pronoun_line.split()
    pos_part = parts[1]
    pos = '~' + pos_part
    return pos


@register.register(['PHN', 'PHN_TONE', 'PHN_TONE_SEP', 'SHENGYUN'], parse_line_methods)
def parse_line_default(pronoun_line, py_type):
    pinyins = parse_pinyin(pronoun_line, py_type)
    punc = parse_punct(pronoun_line)
    return pinyins + punc


def parse_line(pronoun_line, py_type):
    try:
        return parse_line_methods[py_type](pronoun_line, py_type)
    except KeyError:
        raise ValueError('parse_line for [{}] is not implemented'.format(py_type))


@register.register(['PHN', 'PHN_TONE', 'PHN_TONE_SEP', 'SHENGYUN'], parse_sent_methods)
def parse_sent_default(pronoun_lines, py_type, use_head, use_tail):
    if use_head:
        sent_outputs = ['_HEAD']
    else:
        sent_outputs = []
    for line_idx, pronoun_line in enumerate(pronoun_lines):
        if pronoun_line == '' or pronoun_line.startswith('#') or pronoun_line.startswith('['):
            continue
        else:
            line_outputs = parse_line(pronoun_line, py_type)
            if line_idx == len(pronoun_lines) - 1 and line_outputs[-1].startswith('_'):
                line_outputs[-1] += '_E'
            sent_outputs.extend(line_outputs)

    for phn_idx, phn_item in enumerate(sent_outputs):
        try:
            sent_outputs[phn_idx] = final_punc_map[phn_item]
        except KeyError as e:
            pass
    if use_tail:
        sent_outputs.append('_TAIL')
    return sent_outputs


def parse_sent(pronoun_lines, py_type, use_head=True, use_tail=True):
    try:
        return parse_sent_methods[py_type](pronoun_lines, py_type, use_head, use_tail)
    except KeyError:
        raise ValueError('parse_sent for [{}] is not implemented'.format(py_type))
