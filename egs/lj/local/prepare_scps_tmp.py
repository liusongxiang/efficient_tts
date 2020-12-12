import argparse
import os, sys
from tqdm import tqdm
import numpy as np
import random

random.seed(1234)

"""
1. Split dataset into training, dev and eval set.
2. Prepare text, feats.scp, f0.scp, durations.scp
3. Prepare token.list
"""

parser = argparse.ArgumentParser()
parser.add_argument('--feature_root', type=str, help="Root directory of the processed features")
parser.add_argument('--train_set', type=str, default="../dump/train_set")
parser.add_argument('--dev_set', type=str, default="../dump/dev_set")
parser.add_argument('--eval_set', type=str, default="../dump/eval_set")

args = parser.parse_args()

data_root = args.feature_root  # "/home/shaunxliu/data/datasets/LibriTTS"

def read_phnseq_file(phn_f):
    with open(phn_f, 'r') as f:
        phnseq = f.read().strip().split()
    return " ".join(phnseq)

def process_dataset(target_dir, fid_list):
    # target_dir = f'{trans_type}_train_clean460_nodev'
    out_dir = f'../dump/{target_dir}'
    os.makedirs(out_dir, exist_ok=True)

    specs_scp_f = open(f'{out_dir}/specs.scp', 'w')
    
    for fid in tqdm(fid_list):

        spec_path = os.path.abspath(f"{data_root}/{fid}.stft.npy")
        specs_scp_f.write(f"{fid} {spec_path}\n")       
    
    specs_scp_f.close()
     
# Read in all the file ids.
with open(f"{data_root}/fid_list.all", 'r') as f:
    fid_list_all = [l.strip() for l in f]

# 1. Split data into training, dev and test
random.shuffle(fid_list_all)
fid_list_dev = fid_list_all[:100]
fid_list_eval = fid_list_all[100:200]
fid_list_train = fid_list_all[200:]

# 2. Prepare text, feats.scp, f0.scp, durations.scp
print("Prepare dev set.")
process_dataset(target_dir=args.dev_set,
                fid_list=fid_list_dev)
print("Prepare eval set.")
process_dataset(target_dir=args.eval_set,
                fid_list=fid_list_eval)
print("Prepare training set.")
process_dataset(target_dir=args.train_set,
                fid_list=fid_list_train)

# 3. Prepare token.list
# phn_token_list = 'local/durian_preprocess_scripts/text/cn_phn_set_from_txdata.txt'
# with open(phn_token_list, 'r') as f:
    # token_list = [l.strip() for l in f]
# token_list.insert(0, '<unk>')
# token_list.insert(0, '<blank>')
# token_list.append('<sos/eos>')
# with open('dump/token.txt', 'w') as f:
    # f.write('\n'.join(token_list))


