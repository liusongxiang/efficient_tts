import argparse
import os, sys
from tqdm import tqdm
import numpy as np
import glob
import random
random.seed(1234)


"""
1. Split dataset into training, dev and eval set.
2. Prepare text, feats.scp, durations.scp, spkid.scp
3. Prepare token.list
"""


libritts_phnseq_duration_dir = '/home/shaunxliu/data_96/LibriTTS/phnseq_duration_mfa_clean460'
libritts_spk_dvec_dir = '/home/shaunxliu/data_96/LibriTTS/GE2E_spkEmbed_step_5805000_perSpk'
libritts_feature_dir = '/home/shaunxliu/data_96/LibriTTS/vc_features'
# cmuarctic_phnseq_duration_dir = '/home/shaunxliu/data/cmu_arctic/phnseq_duration_mfa'
vctk_phnseq_duration_dir = '/home/shaunxliu/data_96/vctk/phnseq_duration_mfa'
vctk_spk_dvec_dir = '/home/shaunxliu/data_96/vctk/GE2E_spkEmbed_step_5805000_perSpk'
vctk_feature_dir = '/home/shaunxliu/projects/nnsp/egs/voice_transformation/seq2durian_vc_16k/dump/features'


def read_phnseq_file(phn_f):
    with open(phn_f, 'r') as f:
        phnseq = f.read().strip().split()
    return " ".join(phnseq)


# def get_spk2idx_map(
    # # libritts_spkset_file="/home/shaunxliu/data_96/LibriTTS/spk_set.txt",
    # vctk_spkset_file="/home/shaunxliu/data_96/vctk/spk_set.txt",
    # cmuarctic_spkset_file="/home/shaunxliu/data/cmu_arctic/spk_set.txt"
# ):
    # spk_list = []
    # # with open(libritts_spkset_file, 'r') as f:
        # # spk_list.extend([l.strip() for l in f])
    # with open(vctk_spkset_file, 'r') as f:
        # spk_list.extend([l.strip() for l in f])
    # with open(cmuarctic_spkset_file, 'r') as f:
        # spk_list.extend([l.strip() for l in f])
    # print(f"[get_spk2idx_map]: In total {len(spk_list)} speakers.")
    # spk2idx_map = dict(zip(spk_list, range(len(spk_list))))
    # return spk2idx_map

# SPK2IDX = get_spk2idx_map()


def prepare_token_list(phn_set_file, save_fname):
    with open(phn_set_file, 'r') as f:
        token_list = [l.strip() for l  in f]
    token_list.insert(0, '<unk>')
    token_list.insert(0, '<blank>')
    token_list.append('<sos/eos>')
    with open(save_fname, 'w') as f:
        f.write('\n'.join(token_list))


def process_dataset(out_dir, fid_list):
    os.makedirs(out_dir, exist_ok=True)

    feats_scp_f = open(f'{out_dir}/feats.scp', 'w')
    feats_shape_f = open(f'{out_dir}/feats_shape', 'w')
    text_f = open(f'{out_dir}/text', 'w')
    dur_scp_f = open(f'{out_dir}/durations.scp', 'w')
    spk_scp_f = open(f'{out_dir}/spk_dvec.scp', 'w')
    
    for fid in tqdm(fid_list):
        if fid.startswith("p"):
            # vctk
            mel_path = f'{vctk_feature_dir}/{fid}.mel.npy'
            # (T, 80)
            mel = np.load(mel_path)
            mel_path = os.path.abspath(mel_path)
            if len(mel) >= 1000:
                continue
            phnseq_file = f"{vctk_phnseq_duration_dir}/{fid}.phnseq.txt"
            dur_path = f"{vctk_phnseq_duration_dir}/{fid}.dur.npy"

            spkname = fid.split('_')[0]
            spk_dvec_path = f"{vctk_spk_dvec_dir}/{spkname}.npy"
        else:
            # libritts
            mel_path = f'{libritts_feature_dir}/{fid}.mel.npy'
            # (T, 80)
            mel = np.load(mel_path)
            mel_path = os.path.abspath(mel_path)
            if len(mel) >= 1000:
                continue

            phnseq_file = f"{libritts_phnseq_duration_dir}/{fid}.phnseq.txt"
            dur_path = f"{libritts_phnseq_duration_dir}/{fid}.dur.npy"

            spkname = fid.split('_')[0]
            spk_dvec_path = f"{libritts_spk_dvec_dir}/{spkname}.npy"
        try:
            phn_seq = read_phnseq_file(phnseq_file)
        except:
            continue
        text_f.write(f"{fid} {phn_seq}\n")
        dur_path = os.path.abspath(dur_path)
        dur_scp_f.write(f"{fid} {dur_path}\n")
        # spk_id = SPK2IDX[spkname]
        spk_scp_f.write(f"{fid} {spk_dvec_path}\n")
        feats_scp_f.write(f"{fid} {mel_path}\n")
        feats_shape_f.write(f"{fid} {mel.shape[0]}\n")

    feats_scp_f.close()
    feats_shape_f.close()
    text_f.close()
    dur_scp_f.close()
    spk_scp_f.close()
    # spk_scp_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--feature_root', type=str, help="Root directory of the processed features")
    parser.add_argument('--train_set', type=str, default="train_set_vctk_libritts460")
    parser.add_argument('--dev_set', type=str, default="dev_set_vctk")
    parser.add_argument('--eval_set', type=str, default="eval_set_vctk")
    # parser.add_argument('--valid_size', type=int, default=500)
    # parser.add_argument('--test_size', type=int, default=500)

    args = parser.parse_args()
    
    output_dir = "/home/shaunxliu/projects/nnsp/egs/voice_transformation/seq2durian_vc_16k/dump/"

    # target_speakers = ["p225", "p226", "p227", "p228"]
    # target_speakers = ["bdl", "clb", "rms", "slt"]
    # feature_root = args.feature_root
    # valid_size = args.valid_size
    # test_size = args.test_size

    # Read in all the file ids.
    # fid_list_all = [os.path.basename(l).split(".")[0] for l in glob.glob(f"{feature_root}/p*.mel.npy")
                    # if "arctic" not in l]
    # fid_list_all_cmuarctic=[os.path.basename(l).split(".")[0] for l in glob.glob(f"{feature_root}/*.mel.npy")
                    # "arctic" not in l]
    # print(f"[INF0] Glob {len(fid_list_all)} files from vctk.")
    
    # 25 utts for dev, 25 for test and the remainings for training for target
    # speakers.
    # fid_list_trg_spks_train = []
    # fid_list_trg_spks_dev = []
    # fid_list_trg_spks_eval = []
    # for trg_spk in target_speakers:
        # cur_fids = [fid for fid in fid_list_all if trg_spk in fid]
        # random.shuffle(cur_fids)
        # fid_list_trg_spks_dev.extend(cur_fids[:25])
        # fid_list_trg_spks_eval.extend(cur_fids[25:50])
        # fid_list_trg_spks_train.extend(cur_fids[50:])
     
    # # filtering fids: not Indian Speakers and not in target speakers.
    # fid_list_all = [fid for fid in fid_list_all if fid.split("_")[0] in SPK2IDX and not fid.split('_')[0] in target_speakers]
    # print(f"Remaining fids: {len(fid_list_all)}")
    
    # # Add in cmu-arctic fids
    # with open("/home/shaunxliu/data/cmu_arctic/train_fidlist.txt", "r") as f:
        # fid_list_cmu_train = [l.strip() for l in f]
    # with open("/home/shaunxliu/data/cmu_arctic/valid_fidlist.txt", "r") as f:
        # fid_list_cmu_dev = [l.strip() for l in f]
    # with open("/home/shaunxliu/data/cmu_arctic/test_fidlist.txt", "r") as f:
        # fid_list_cmu_eval = [l.strip() for l in f]


    # # 1. Split data into training, dev and test
    # random.shuffle(fid_list_all)
    # fid_list_dev = fid_list_all[:valid_size] + fid_list_trg_spks_dev + fid_list_cmu_dev
    # fid_list_eval = fid_list_all[valid_size:valid_size+test_size] + fid_list_trg_spks_eval + fid_list_cmu_eval
    # fid_list_train = fid_list_all[valid_size+test_size:] + fid_list_trg_spks_train + fid_list_cmu_train
    
    # 2. Prepare token list
    # token_list_fname = os.path.dirname(feature_root) + "/tokens.txt"
    # phn_set_file = "/home/shaunxliu/data_96/vctk/phone_set.txt"
    # prepare_token_list(phn_set_file, token_list_fname)
    
    fidlist_dir = "/home/shaunxliu/data_96/vctk/fidlists/"
    with open(f"{fidlist_dir}/train_fidlist.txt", 'r') as f:
        fid_list_train = [l.strip() for l in f]
    with open(f"{fidlist_dir}/dev_fidlist.txt", 'r') as f:
        fid_list_dev = [l.strip() for l in f]
    with open(f"{fidlist_dir}/eval_fidlist.txt", 'r') as f:
        fid_list_eval = [l.strip() for l in f]
    print("train: ", len(fid_list_train))
    print("dev: ", len(fid_list_dev))
    print("eval: ", len(fid_list_eval))
    # sys.exit()
    # 3. Prepare text, feats.scp, spkid.scp
    print("Prepare dev set.")
    process_dataset(os.path.join(output_dir, args.dev_set),
                    fid_list=fid_list_dev,
                    )
    print("Prepare eval set.")
    process_dataset(os.path.join(output_dir, args.eval_set),
                    fid_list=fid_list_eval,
                    )
    print("Prepare training set.")
    process_dataset(os.path.join(output_dir, args.train_set),
                    fid_list=fid_list_train,
                    )


