#!/bin/bash

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=0

# basic settings
stage=3        # stage to start
stop_stage=3   # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction
corpus_name=lj

conf=./conf/efficient_tts_cnn_phnseq.v1.yaml

# Specify the data directory 
data_dir=/home/shaunxliu/data_96/LJSpeech-1.1

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training (e.g. <path>/<to>/checkpoint-10000steps.pkl)

# decoding related setting
checkpoint="./exp/lj_efficient_tts_cnn_phnseq.v1/checkpoint-105000steps.pkl" # checkpoint path to be used for decoding
              # if not provided, the latest one will be used
              # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ -z "${tag}" ]; then
    expdir="exp/${corpus_name}_$(basename "${conf}" .yaml)"
else
    expdir="exp/${corpus_name}_${tag}"
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    #echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
    #${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
    nntts-inference \
      --test_fid_scp "/home/shaunxliu/data/LJSpeech-1.1/nv_taco2_filelists/ljs_audio_phnseq_test_filelist.txt" \
      --checkpoint "${checkpoint}" \
      --outdir "${outdir}" \
      --verbose "${verbose}"
          
    echo "Successfully finished decoding."
fi
echo "Finished."
