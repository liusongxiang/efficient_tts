#!/bin/bash

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;

export CUDA_VISIBLE_DEVICES=3

# basic settings
stage=2        # stage to start
stop_stage=2   # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction
corpus_name=lj

#conf=./conf/efficient_tts_cnn_phnseq_defaultConf.v1.yaml
conf=./conf/efficient_tts_cnn_phnseq.debug.yaml

# Specify the data directory 
data_dir=/home/shaunxliu/data/LJSpeech-1.1

# training related setting
tag=""     # tag for directory to save model
resume=""
#resume="./exp/lj_efficient_tts_cnn.v1/checkpoint-2000steps.pkl"  # checkpoint path to resume training (e.g. <path>/<to>/checkpoint-10000steps.pkl)

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
    echo "Stage 1: Data Preprocessing"
    local/data.sh --data_config `pwd`/conf/data_preprocess.yaml \
                  --data_root "${data_dir}" \
                  --scp_dir `pwd`/dump/scps \
                  --feature_dir `pwd`/dump/features
fi


if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    #cp "${dumpdir}/${train_set}/stats.${stats_ext}" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        train="python -m nntts.distributed.launch --nproc_per_node ${n_gpus} -c nntts-train"
    else
        train="nntts-train"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    #${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --config "${conf}" \
            --train_fid_scp "/home/shaunxliu/data/LJSpeech-1.1/nv_taco2_filelists/ljs_audio_phnseq_train_filelist.txt" \
            --dev_fid_scp "/home/shaunxliu/data/LJSpeech-1.1/nv_taco2_filelists/ljs_audio_phnseq_val_filelist.txt" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/wav/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${dev_set}" "${eval_set}"; do
    (
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        #${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            nntts-decode \
                --dumpdir "${dumpdir}/${name}/norm" \
                --checkpoint "${checkpoint}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    ) &
    pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    echo "Successfully finished decoding."
fi
echo "Finished."
