import numpy as np
import sys
import os
import re
from os import walk
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import cpu_count

def extract_phn_timestamp(label_lines):
    phn_timestamp = []
    groups = []
    sub_group = []
    try:
        for (i, line) in enumerate(label_lines):
            parts = line.split()
            s = parts[0]
            e = parts[1]
            full_label = parts[2]
            phn = re.match('-(\w+)+', full_label).group(1)
            if i % 5 == 0:
               groups.append(sub_group)
               sub_group = []
            sub_group.append([phn, int(s), int(e), full_label])
        groups.append(sub_group)
        groups = groups[1:]
        
        new_lines = []
        for group in groups:
            if not (group[0][0] == group[1][0] == group[2][0] == group[3][0] == group[4][0]):
                import pdb; pdb.set_trace()
                raise ValueError('PHN in group mismatch')
            phn = group[0][0]
            ns = group[0][1]
            ne = group[-1][2]
            full_label = group[0][3]
            new_lines.append([phn, int(ns), int(ne), full_label])   
        return new_lines
    except:
        return None

def parse_label(in_lab_file):
    try:
        with open(in_lab_file, 'r', encoding='utf-8') as f:
            label_lines = [l.strip() for l in f.readlines()]
        new_lines = extract_phn_timestamp(label_lines)
        return new_lines 
    except:
        return None

def process_one_file(in_lab_file, out_fname):
    new_lines = parse_label(in_lab_file)
    
    if new_lines is not None:
        with open(out_fname, 'w', encoding='utf-8') as f:
            for label in new_lines:
                f.write(' ')
                f.write(str(label[1]))
                f.write(' ')
                f.write(str(label[2]))
                f.write(' ')
                f.write(label[3])
                f.write("\n")
    return 1

if __name__ == '__main__':

    # load_path = sys.argv[1]
    # out_path = sys.argv[2]
    
    raw_label_path = '/home/shaunxliu/data/data_50/tts_align_labels'
    out_path = '/home/shaunxliu/data/features/tx_full_with_dur'
    os.makedirs(out_path, exist_ok=True)
    
    spk_folder_list = os.listdir(raw_label_path)
    print(f'[INFO] In total, find {len(spk_folder_list)} spk folders.')
    
    for spk in spk_folder_list:
        if 'king_f47' in spk: continue
        print(f'[INFO] Processing {spk}')
        load_path = os.path.join(raw_label_path, spk)
        cur_out_path = os.path.join(out_path, spk)
        os.makedirs(cur_out_path, exist_ok=True)
        filenames = []
        for (_, _, f) in walk(load_path):
          filenames.extend(f)
        
        future = []
        executor = ProcessPoolExecutor(40)
        for i in filenames:
            in_fname = os.path.join(load_path, i)
            out_fname = os.path.join(cur_out_path, i)   
            future.append(executor.submit(process_one_file, in_fname, out_fname))
        wait(future)
        num = sum([f.result() for f in future])
        print(f'\t {num} files.')
        executor.shutdown()

        # for i in filenames:
           # new_labels = parse_label(os.path.join(load_path, i)) 
           # if new_lines is not None:
               # with open(os.path.join(out_path,i), 'w', encoding='utf-8') as f:
                  # for label in new_labels:
                      # f.write(' ')
                      # f.write(str(label[1]))
                      # f.write(' ')
                      # f.write(str(label[2]))
                      # f.write(' ')
                      # f.write(label[3])
                      # f.write("\n")
            
