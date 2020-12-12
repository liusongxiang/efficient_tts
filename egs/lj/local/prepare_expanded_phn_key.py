import numpy as np
import glob
from tqdm import tqdm
import os, sys
sys.path.append("..")
from dump.key_to_freq_map import key_to_frequency


## Prepare key-to-idx dict
with open("../dump/key_set.txt", 'r') as f:
# with open("/home/shaunxliu/projects/nnsp/egs/svs/guiniang_ywdata/dump/key_set.txt", 'r') as f:
    key_set = [l.strip() for l in f]
key_to_idx_map = dict(zip(key_set, range(len(key_set))))

## Used by find_nearest_key
frequency_to_key = {v:k for k, v in key_to_frequency.items()}
frequency_array = np.array(list(key_to_frequency.values()))

feature_dir = "../dump/features"

## Collect f0 files
f0_file_list = glob.glob(f"{feature_dir}/*.f0.npy")
print(f"Get {len(f0_file_list)} f0 files.")

def read_phnlist(phn_file):
    """Read phn_tone sequence."""
    with open(phn_file, 'r') as f:
        phn_list = f.read().strip().split()
    return phn_list

def find_nearest_key(f0_value):
    """Mapping real-value f0 to its nearest key according to frequency value."""
    idx = (np.abs(f0_value - frequency_array)).argmin()
    return frequency_to_key[frequency_array[idx]]

## Generate:
# 1. expanded phn_tone sequence.
# 2. categorical f0s in key form, save as npy
# 3. expanded phn_id sequence
# 4. expanded phnid+key sequence
for f0_file in tqdm(f0_file_list):
    uttid = os.path.basename(f0_file).split('.')[0]
    phn_file = f"{feature_dir}/{uttid}.phntone.txt"
    dur_file = f"{feature_dir}/{uttid}.dur.npy"
    ## Read in data
    f0 = np.load(f0_file)
    phn_list = read_phnlist(phn_file)
    dur = np.load(dur_file)
    
    ## Make f0 has proper length
    dur_sum = np.sum(dur)
    pad_num = dur_sum - len(f0)
    if pad_num > 0:
        f0 = np.pad(f0, (0, pad_num), mode="constant", constant_values=f0[-1])
    elif pad_num < 0:
        f0 = f0[:pad_num]
    assert len(f0) == np.sum(dur), "error"
    
    ## Categorize pitch to the nearest key note
    key_list = []
    for f0_value in f0:
        key = find_nearest_key(f0_value)
        key_list.append(key)
    ## Expanded phn_tone sequence
    phn_tone_list_expanded = []
    for phn, d in zip(phn_list, dur):
        phn_repeat = np.repeat([phn], d).tolist()
        phn_tone_list_expanded.extend(phn_repeat)
    ## Expanded phnid+key sequence 
    phnid_key_list = []
    for phn, key in zip(phn_tone_list_expanded, key_list):
        if phn[-1].isnumeric():
            phnid_key = phn[:-1] + '_' + key
        else:
            phnid_key = phn + '_' + key
        phnid_key_list.append(phnid_key)
    ## Expanded phn_id sequence
    expanded_phn_id_list = []
    for phn_tone in phn_tone_list_expanded:
        if phn_tone[-1].isnumeric():
            phn_id = phn_tone[:-1]
        else:
            phn_id = phn_tone
        expanded_phn_id_list.append(phn_id)
    ## Convert string key to idx and save as numpy array
    key_int_list = [key_to_idx_map[key] for key in key_list]
    key_int_list = np.asarray(key_int_list)
    np.save(f"{feature_dir}/{uttid}.key.npy", key_int_list, allow_pickle=False)
    
    with open(f"{feature_dir}/{uttid}.expandphntone.txt", 'w') as f:
        f.write(" ".join(phn_tone_list_expanded))
    with open(f"{feature_dir}/{uttid}.expandphnid.txt", 'w') as f:
        f.write(" ".join(expanded_phn_id_list))
    with open(f"{feature_dir}/{uttid}.expandphnkey.txt", 'w') as f:
        f.write(" ".join(phnid_key_list))
        



