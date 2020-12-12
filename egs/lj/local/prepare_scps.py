import argparse
import os, sys
import shutil
from tqdm import tqdm
import numpy as np
import glob
import random
random.seed(1234)


"""
1. Prepare spkid.scp
2. Split dataset into training, dev and eval set.
3. Prepare token.list
"""

def prepare_spkset_file(fidlist):
    spk_list = list(set([fid.split('_')[0] for fid in fidlist]))
    spk_list.sort()
    return spk_list


def prepare_token_list(phn_set_file, save_fname):
    with open(phn_set_file, 'r') as f:
        token_list = [l.strip() for l  in f]
    token_list.insert(0, '<unk>')
    token_list.insert(0, '<blank>')
    token_list.append('<sos/eos>')
    with open(save_fname, 'w') as f:
        f.write('\n'.join(token_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_root', type=str, help="Root directory of the processed features")
    parser.add_argument('--scp_dir', type=str, help="Text files output directory.")
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--valid_size', type=int, default=200)
    parser.add_argument('--test_size', type=int, default=200)
    parser.add_argument('--spkset_file', type=str, default="None")
    parser.add_argument('--phnset_file', type=str, required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.scp_dir, exist_ok=True)

    # Read in all the file ids.
    fid_list_all = [os.path.basename(l).split(".")[0] for l in glob.glob(f"{args.feature_root}/p*.mel.npy")]
    print(f"[INF0] Glob {len(fid_list_all)} files from vctk.")
    
    # 1. Prepare speaker list file
    if args.spkset_file == "None":
        spk_list = prepare_spkset_file(fid_list_all)
        with open(f"{args.scp_dir}/speaker_list", 'w') as f:
            f.write("\n".join(spk_list))
    else:
        shutil.copy(f"{args.data_root}/{args.spkset_file}", f"{args.scp_dir}/speaker_list")
    
    # 2. Split data into training, dev and test
    random.shuffle(fid_list_all)
    fid_list_valid = fid_list_all[:args.valid_size]
    fid_list_test = fid_list_all[args.valid_size:args.valid_size+args.test_size]
    fid_list_train = fid_list_all[args.valid_size+args.test_size:]
    with open(f"{args.scp_dir}/train_fidlist", 'w') as f:
        f.write("\n".join(fid_list_train))
    with open(f"{args.scp_dir}/valid_fidlist", 'w') as f:
        f.write("\n".join(fid_list_valid))
    with open(f"{args.scp_dir}/test_fidlist", 'w') as f:
        f.write("\n".join(fid_list_test))

    # 3. Prepare token list
    token_list_fname = f"{args.scp_dir}/tokens.txt"
    prepare_token_list(f"{args.data_root}/{args.phnset_file}", token_list_fname)
