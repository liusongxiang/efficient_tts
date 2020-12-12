#!/bin/bash

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=0

# basic settings
stage=1        # stage to start
stop_stage=1   # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction
corpus_name=lj

conf=./conf/efficient_tts_cnn_phnseq_noDropout.v1.yaml

# Specify the data directory 
data_dir=path/to/LJSpeech-1.1

# training related setting
tag=""     # tag for directory to save model
resume=""

# decoding related setting
checkpoint="" # checkpoint path to be used for decoding
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

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    #cp "${dumpdir}/${train_set}/stats.${stats_ext}" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m nntts.distributed.launch --nproc_per_node ${n_gpus} -c nntts-train"
    else
        train="nntts-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${train} \
        --config "${conf}" \
        --train_fid_scp "nv_taco2_filelists/ljs_audio_phnseq_train_filelist.txt" \
        --dev_fid_scp "nv_taco2_filelists/ljs_audio_phnseq_val_filelist.txt" \
        --outdir "${expdir}" \
        --resume "${resume}" \
        --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Synthesis"
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    nntts-inference \
      --test_fid_scp "nv_taco2_filelists/ljs_audio_phnseq_test_filelist.txt" \
      --checkpoint "${checkpoint}" \
      --outdir "${outdir}" \
      --verbose "${verbose}"
          
    echo "Successfully finished Synthesis."
fi

echo "Finished."
