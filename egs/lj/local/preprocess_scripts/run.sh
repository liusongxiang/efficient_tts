
python preprocess.py --base_dir /home/shaunxliu/data/datasets/LibriTTS \
                     --output phnseq_duration \
                     --fid_list dev_test.filelist \
                     --trim_silence False \
                     --use_seg True \
                     --num_workers 20 \
                     --phn2idx text/phn_idx.map
