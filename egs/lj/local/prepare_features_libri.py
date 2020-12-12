import configargparse
import os, sys
import glob
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

from preprocess_scripts.audio import load_wav, spectrogram, melspectrogram
# from preprocess_scripts.f0 import compute_f0_from_wav


"""
Prepare melspectrograms.
"""


def str2bool(v):
   if v.lower() in ('yes', 'true', 't', 'y', '1'):
       return True
   elif v.lower() in ('no', 'false', 'f', 'n', '0'):
       return False
   else:
       raise argparse.ArgumentTypeError('Boolean value expected.')


@dataclass
class AudioConfig: 
    num_mels: int = 80
    num_freq: int = 1025
    sample_rate: int = 16000
    frame_length_ms: int = 50
    frame_shift_ms: int = 10
    pre_emphasis: float = 0.97
    fmin: int = 40
    min_level_db: int = -100
    ref_level_db: int = 20


def _process_utterance(
    out_dir, 
    fid, 
    wav_path,
    audio_config
):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
        out_dir: The directory to write the spectrograms into
        fid: utterance id.
        wav_path: Path to the audio file containing the speech input
        audio_config:
    '''
    # Load the audio to a numpy array:
    wav = load_wav(wav_path, audio_config.sample_rate)

    # Compute a mel-scale spectrogram for wavenet, wavernn 
    mel_spectrogram = melspectrogram(wave=wav,
                                     sample_rate=audio_config.sample_rate,
                                     num_freq=audio_config.num_freq,
                                     num_mels=audio_config.num_mels,
                                     frame_length_ms=audio_config.frame_length_ms,
                                     frame_shift_ms=audio_config.frame_shift_ms,
                                     pre_emphasis=audio_config.pre_emphasis,
                                     fmin=audio_config.fmin,
                                     min_level_db=audio_config.min_level_db,
                                     ref_level_db=audio_config.ref_level_db
                                     ).astype(np.float32)
    mel_spectrogram = mel_spectrogram.T  # -> [T, num_mels]
    # stft = spectrogram(wave=wav,
                       # sample_rate=audio_config.sample_rate,
                       # num_freq=audio_config.num_freq,
                       # frame_length_ms=audio_config.frame_length_ms,
                       # frame_shift_ms=audio_config.frame_shift_ms,
                       # pre_emphasis=audio_config.pre_emphasis,
                       # min_level_db=audio_config.min_level_db,
                       # ref_level_db=audio_config.ref_level_db
                       # ).astype(np.float32)
    # stft = stft.T  # -> [T, num_freq]
    
    np.save(f'{out_dir}/{fid}.mel.npy', mel_spectrogram, allow_pickle=False)
    return len(mel_spectrogram), fid


def preprocess_run(args):
    # out_dir = os.path.join(args.base_dir, args.output)
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Dirs
    libri_mfa_dir = "/home/shaunxliu/data_96/LibriSpeech/MFA_labs"
    libri_wav_dir = "/home/shaunxliu/data_96/LibriSpeech/LibriSpeech"
    
    # Get file id list
    fid_list = []
    fid_list.extend(glob.glob(f"{libri_mfa_dir}/*/*/*/*.lab"))
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


    executor = ProcessPoolExecutor(args.num_workers)
    futures = []
    
    for lab_fid in fid_list:
        split_name = lab_fid.split("/")[-4]
        fid = os.path.basename(lab_fid).split(".")[0]
        parts = fid.split("-")
        # print(parts)
        spk_name = parts[0]
        chap_name = parts[1]
        wave_path = f"{libri_wav_dir}/{split_name}/{spk_name}/{chap_name}/{fid}.flac"
        if not os.path.exists(wave_path):
            print(wave_path)
            continue
        futures.append(executor.submit(partial(_process_utterance, 
                                               out_dir, fid, wave_path, audio_config)))
    results =  [future.result() for future in tqdm(futures)]


def get_parser():
    """Get argument parser."""
    parser = configargparse.ArgumentParser(
        description="Data preprocessing",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', is_config_file=True, help="Abspath of config YAML file.")
    parser.add_argument('--output_dir', default=None, required=True, help="Output directory saving the features.")
    
    group = parser.add_argument_group("Data preprocess related")
    group.add_argument('--trim_silence', type=str2bool, default=False)
    group.add_argument('--num_mels', type=int, default=80)
    group.add_argument('--num_freq', type=int, default=1025)
    group.add_argument('--sample_rate', type=int, default=16000)
    group.add_argument('--frame_length_ms', type=int, default=50)
    group.add_argument('--frame_shift_ms', type=int, default=10)
    group.add_argument('--pre_emphasis', type=float, default=0.97)
    group.add_argument('--fmin', type=int, default=40)
    group.add_argument('--min_level_db', type=int, default=-100)
    group.add_argument('--ref_level_db', type=int, default=20)

    group.add_argument('--num_workers', type=int, default=cpu_count())

    return parser 


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    preprocess_run(args)


if __name__ == "__main__":
    main()
