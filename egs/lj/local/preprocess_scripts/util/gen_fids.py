import glob
import os
from tqdm import tqdm 

corpus_dir = '/home/shaunxliu/data/data_50/tts_align_labels/'
out_dir = '../filelists'

spk_list = os.listdir(corpus_dir)
print(f'[INFO] Find {len(spk_list)} spk folders.')

all_lab_list = []

for spk in spk_list:
    cur_dir = os.path.join(corpus_dir, spk)
    lab_list = [os.path.basename(i).split('.')[0] for i in glob.glob(f"{cur_dir}/*.lab")]
    print(f'{spk}: {len(lab_list)} label files.')
    all_lab_list.extend(lab_list)
    with open(f"{out_dir}/{spk}_fids.txt", 'w') as f:
        f.write('\n'.join(lab_list))
        f.write('\n')
 
print(f'In total: got {len(all_lab_list)} label files.')
with open(f"{out_dir}/all_fids.txt", 'w') as f:
    f.write('\n'.join(all_lab_list))
    f.write('\n')


