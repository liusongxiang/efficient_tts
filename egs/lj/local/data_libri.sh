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

stage=1
stop_stage=2

train_set=
dev_set=
eval_set=
data_config=
data_root=
output_dir=

log "$0 $*"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

cwd=$(pwd)

## Procedures:
# 1: Prepare token_list, Mel-sepctrograms, phoneme sequence, duration information
# 2: Prepare continuous F0
# 3: Split dataset into train_set, dev_set and eval_set. Then generate scps.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  log "Stage 1: prepare mel-spectrograms."
  cd local
  python prepare_features_libri.py --output_dir ${output_dir} \
                             --config ${data_config} 
  cd ..
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  log "Stage 2: Prepare training set, dev set and eval set scps."
  #mkdir -p dump/${train_set} dump/${dev_set} dump/${eval_set}
  python local/prepare_scps_libri.py --feature_root ${output_dir} \
                               #--train_set ${train_set} \
                               #--dev_set ${dev_set} \
                               #--eval_set ${eval_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
  


  
  








