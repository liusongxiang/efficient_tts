import configargparse
import os, sys
from dataclasses import dataclass
from multiprocessing import cpu_count
from tqdm import tqdm
import feature_extract
from collections import OrderedDict
from text import parse_pronounce


def str2bool(v):
   if v.lower() in ('yes', 'true', 't', 'y', '1'):
       return True
   elif v.lower() in ('no', 'false', 'f', 'n', '0'):
       return False
   else:
       raise argparse.ArgumentTypeError('Boolean value expected.')


@dataclass
class F0Config:
    method: str
    use_reaper: bool   # False: rapt&melodia for singing, True: reaper for speech
    extract_FRAME_SHIFT_MS: int = 5
    save_FRAME_SHIFT_MS: int = 10


@dataclass
class AudioConfig: 
    num_mels: int = 80
    num_freq: int = 1025
    sample_rate: int = 24000
    frame_length_ms: int = 50
    frame_shift_ms: int = 10
    pre_emphasis: float = 0.97
    fmin: int = 40
    min_level_db: int = -100
    ref_level_db: int = 20


def preprocess_run(args):
    # out_dir = os.path.join(args.base_dir, args.output)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Dirs
    wav_dir = f"{args.data_root}/wav_mono_24k_16b_norm-6db-crop_sil"
    pos_prosody_phn_dir = f"{args.data_root}/pos-prosody-phn"
    full_label_dir = f"{args.data_root}/song_guiniang-flab-crop_sil"
    
    # Load phone map
    phn2idx_map = load_phn_idx_map(args.phn2idx)
    expand_phn2idx = False
    
    # Get file id list
    fid_list = [i[:-4] for i in os.listdir(full_label_dir) if i.endswith('.lab')]  
    print(f"Get {len(fid_list)} files.")

    # Audio config
    audio_config = AudioConfig(num_mels=args.num_mels,
                               num_freq=args.num_freq,
                               sample_rate=args.sample_rate,
                               frame_length_ms=args.frame_length_ms,
                               frame_shift_ms=args.frame_shift_ms,
                               pre_emphasis=args.pre_emphasis,
                               fmin=args.fmin,
                               min_level_db=args.min_level_db,
                               ref_level_db=args.ref_level_db
                               )
    print("Audio config: \n\t", audio_config)

    # f0 config
    f0_config = F0Config(method=args.method, 
                         use_reaper=args.use_reaper, 
                         extract_FRAME_SHIFT_MS=args.extract_FRAME_SHIFT_MS,
                         save_FRAME_SHIFT_MS=args.save_FRAME_SHIFT_MS,
                         )
    print("F0 config: \n\t", f0_config)

    metadata, phn2idx_map = feature_extract.build_from_path(
                                fid_list=fid_list,
                                base_dir=args.data_root,
                                wav_dir=wav_dir,
                                full_label_dir=full_label_dir,
                                pos_prosody_phn_dir=pos_prosody_phn_dir,
                                out_dir=out_dir,
                                num_workers=args.num_workers,
                                phn2idx_map=phn2idx_map,
                                audio_config=audio_config,
                                f0_config=f0_config,
                                tqdm=tqdm,
                                trim_silence=args.trim_silence,
                                expand_phn2idx=expand_phn2idx,
                                use_seg=args.use_seg,
                                use_head=args.use_head,
                                use_tail=args.use_tail,
                                py_type=args.py_type
                                )
    # write_metadata(metadata, out_corpus_dir, corpus_id)

    phn2idx_map_file = os.path.join(out_dir, 'phn_idx.map')
    with open(phn2idx_map_file, 'wt', encoding='utf-8') as F:
        for key in phn2idx_map.keys():
            print("{} {}".format(key, phn2idx_map[key]), file=F)
    fid_list_processed = [tup[1] for tup in metadata]
    fid_list_processed.sort()
    with open(f"{out_dir}/fid_list.all", 'w') as f:
        f.write('\n'.join(fid_list_processed))


def get_parser():
    """Get argument parser."""
    parser = configargparse.ArgumentParser(
        description="Data preprocessing",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', is_config_file=True, help="Abspath of config YAML file.")
    parser.add_argument('--data_root', default=None, required=True, help="Base data path.")
    parser.add_argument('--train_set', default=None, required=True, help="Training set name.")
    parser.add_argument('--dev_set', default=None, required=True, help="Development set name.")
    parser.add_argument('--eval_set', default=None, required=True, help="Test set name.")
    parser.add_argument('--output_dir', default=None, required=True, help="Output directory saving the features.")
    parser.add_argument('--phn2idx', default='text/phn_idx.map')
    
    group = parser.add_argument_group("Data preprocess related")
    group.add_argument('--py_type', type=str, default="PHN_TONE", help="Pinyin type.")
    group.add_argument('--use_head', type=str2bool, default=True, help="Use _HEAD as one symbol.")
    group.add_argument('--use_tail', type=str2bool, default=False, help="Use _TAIL as one symbol.")
    group.add_argument('--use_seg', type=str2bool, default=True, help="Use segment symbols #1 #2 ...")
    group.add_argument('--trim_silence', type=str2bool, default=False)
    group.add_argument('--num_mels', type=int, default=80)
    group.add_argument('--num_freq', type=int, default=1025)
    group.add_argument('--sample_rate', type=int, default=24000)
    group.add_argument('--frame_length_ms', type=int, default=50)
    group.add_argument('--frame_shift_ms', type=int, default=10)
    group.add_argument('--pre_emphasis', type=float, default=0.97)
    group.add_argument('--fmin', type=int, default=40)
    group.add_argument('--min_level_db', type=int, default=-100)
    group.add_argument('--ref_level_db', type=int, default=20)
    # F0 related
    group.add_argument('--method', type=str, default='melodia')
    group.add_argument('--use_reaper', type=str2bool, default=False)
    group.add_argument('--extract_FRAME_SHIFT_MS', type=int, default=5)
    group.add_argument('--save_FRAME_SHIFT_MS', type=int, default=10)

    group.add_argument('--num_workers', type=int, default=cpu_count())

    return parser 


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    preprocess_run(args)


if __name__ == "__main__":
    main()
