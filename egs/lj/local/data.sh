#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail


log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
  }
SECONDS=0

stage=2
stop_stage=2

data_config=
data_root=
scp_dir=
feature_dir=

wav_folder=wav48_trimSil
mfa_folder=phnseq_duration_mfa
num_valid_utts=200
num_test_utts=200
spkset_file=None
phnset_file=phone_set.txt


log "$0 $*"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

cwd=$(pwd)

## Procedures:
# 1: Prepare token_list, Mel-sepctrograms, phoneme sequence, duration information
# 2: Split dataset into train_set, dev_set and eval_set. Then generate fidlists.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: prepare mel-spectrograms."
  cd local
  python prepare_features.py --output_dir ${feature_dir} \
                             --config ${data_config} \
                             --data_root ${data_root} \
                             --wav_folder ${wav_folder} \
                             --mfa_folder ${mfa_folder}
  cd ..
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: Prepare training set, dev set and eval set scps. Also speaker_list and phone-set list"
  python local/prepare_scps.py --feature_root ${feature_dir} \
                               --scp_dir ${scp_dir} \
                               --valid_size ${num_valid_utts} \
                               --test_size ${num_test_utts} \
                               --data_root ${data_root} \
                               --spkset_file ${spkset_file} \
                               --phnset_file ${phnset_file}
                               
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
  


  
  








