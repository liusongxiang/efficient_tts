import argparse
import os, sys
from tqdm import tqdm
import numpy as np
import glob
import random
random.seed(1234)


"""
1. Prepare text, feats.scp, durations.scp, spkid.scp
2. Prepare token.list
"""

libri_root_dir = "/home/shaunxliu/data_96/LibriSpeech"
libri_mfa_lab_dir = "/home/shaunxliu/data_96/LibriSpeech/MFA_labs"


def read_phnbnd_file(fname):
    with open(fname, 'r') as f:
        lines = [l.strip().split() for l in f]
    return lines

def get_phnseq(lab_path):
    lines = read_phnbnd_file(lab_path)
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    phnseq_list = [l[2] for l in lines]
    return " ".join(phnseq_list)

def prepare_token_list(phn_set_file, save_fname):
    with open(phn_set_file, 'r') as f:
        token_list = [l.strip() for l  in f]
    # token_list.insert(0, '<unk>')
    # token_list.insert(0, '<blank>')
    # token_list.append('<sos/eos>')
    with open(save_fname, 'w') as f:
        f.write('\n'.join(token_list))

def process_train_set(src_trg_list, feature_root):
    out_dir = f"dump/{src_trg_list[0][1]}"
    os.makedirs(out_dir, exist_ok=True)
    feats_scp_f = open(f'{out_dir}/feats.scp', 'w')
    feats_shape_f = open(f'{out_dir}/feats_shape', 'w')
    text_f = open(f'{out_dir}/text', 'w')
    for src_trg in src_trg_list:
        split_name = src_trg[0]
        print("Processing: ", split_name)
        lab_file_list = glob.glob(f"{libri_mfa_lab_dir}/{split_name}/*/*/*.lab")
        for lab_fname in tqdm(lab_file_list):
            fid = os.path.basename(lab_fname)[:-4]
            mel_path = f'{feature_root}/{fid}.mel.npy'
            if not os.path.exists(mel_path):
                continue
            # (T, 80)
            mel = np.load(mel_path)
            mel_path = os.path.abspath(mel_path)
            phnseq = get_phnseq(lab_fname)
            feats_scp_f.write(f"{fid} {mel_path}\n")
            feats_shape_f.write(f"{fid} {mel.shape[0]}\n")
            text_f.write(f"{fid} {phnseq}\n")

    feats_scp_f.close()
    feats_shape_f.close()
    text_f.close()
    
def process_one_pair(src_trg, feature_root):
    out_dir = f"dump/{src_trg[1]}"
    os.makedirs(out_dir, exist_ok=True)
    split_name = src_trg[0]
    print("Processing: ", split_name)
    feats_scp_f = open(f'{out_dir}/feats.scp', 'w')
    feats_shape_f = open(f'{out_dir}/feats_shape', 'w')
    text_f = open(f'{out_dir}/text', 'w')

    lab_file_list = glob.glob(f"{libri_mfa_lab_dir}/{split_name}/*/*/*.lab")
    for lab_fname in tqdm(lab_file_list):
        fid = os.path.basename(lab_fname)[:-4]
        mel_path = f'{feature_root}/{fid}.mel.npy'
        if not os.path.exists(mel_path):
            continue
        # (T, 80)
        mel = np.load(mel_path)
        mel_path = os.path.abspath(mel_path)
        phnseq = get_phnseq(lab_fname)
        feats_scp_f.write(f"{fid} {mel_path}\n")
        feats_shape_f.write(f"{fid} {mel.shape[0]}\n")
        text_f.write(f"{fid} {phnseq}\n")

    feats_scp_f.close()
    feats_shape_f.close()
    text_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_root', type=str, help="Root directory of the processed features")
    parser.add_argument('--train_set', type=str)
    parser.add_argument('--dev_set', type=str)
    # parser.add_argument('--eval_set', type=str)

    args = parser.parse_args()

    # split_names = ["train-clean-100", "dev-clean", "test-clean", "train-clean-360", "train-other-500",
                   # "dev-other", "test-other"]
    
    src_trg_pair = [("train-clean-100", "train_libri_960"), ("train-clean-360", "train_libri_960"),
                    ("train-other-500", "train_libri_960"), ("dev-clean", "dev_libri"), 
                    ("dev-clean", "dev_clean_libri"),
                    ("dev-other", "dev_other_libri"), ("test-clean", "test_clean_libri"), 
                    ("test-other", "test_other_libri")]

    feature_root = args.feature_root
    
    # 1. Prepare text, feats.scp, spkid.scp
    process_train_set(src_trg_pair[:3], feature_root)

    for src_trg in src_trg_pair[3:]:
        process_one_pair(src_trg, feature_root)

    # 2. Prepare token list
    # token_list_fname = os.path.dirname(feature_root) + "/tokens.txt"
    # phn_set_file = "/home/shaunxliu/data/datasets/vctk/phone_set.txt"
    # prepare_token_list(phn_set_file, token_list_fname)



