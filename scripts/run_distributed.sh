#!/usr/bin/env bash
#
# Emulate N nodes × G GPUs-per-node on localhost
#
# Usage: ./run_distributed.sh [NUM_NODES] [GPUS_PER_NODE] [HOST] [BASE_PORT]
# Example: ./run_distributed.sh 2 4 127.0.0.1 12355

NUM_NODES=2
GPUS_PER_NODE=1
HOST="127.0.0.1"
BASE_PORT="12356"
DEVICE_OFFSET=2
if [[ $1 == "--simulate" ]]
then
shift 1
export SIMULATE="echo"
fi


if [[ $1 == "--rank_zero_adapt_pre" ]]
then
shift 1
RANK_ZERO_ADAPT_PRE=$1
shift 1
fi


CMD=$1
shift 1

if [[ $1 == "--rank_zero_adapt" ]]
then
shift 1
RANK_ZERO_ADAPT=$1
shift 1
fi

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_ALGO=Ring
export NCCL_COLLNET_ENABLE=0
export NCCL_NCHANNELS=1
export NCCL_BUFFSIZE=262144
export NCCL_NTHREADS=1
export NCCL_SOCKET_NTHREADS=1
export NCCL_LAUNCH_MODE=GROUP



echo "Launching ${TOTAL_PROCS} processes on ${NUM_NODES} virtual nodes…"

for (( NODE_RANK=1; NODE_RANK<NUM_NODES; NODE_RANK++ )); do
    
    echo CUDA_VISIBLE_DEVICES\=$(seq -s, "$(( NODE_RANK * GPUS_PER_NODE + DEVICE_OFFSET ))" "$(( NODE_RANK * GPUS_PER_NODE + GPUS_PER_NODE + DEVICE_OFFSET - 1 ))") \
      $SIMULATE python $CMD --dist_info "${HOST}:${BASE_PORT},${NODE_RANK},${NUM_NODES}" $@
    (
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=^lo
    export NCCL_SHM_DISABLE=1
    export NCCL_P2P_DISABLE=1
    export NCCL_P2P_LEVEL=SYS
    export NCCL_ALGO=Ring
    export NCCL_COLLNET_ENABLE=0
    export NCCL_NCHANNELS=1
    export NCCL_BUFFSIZE=262144
    export NCCL_NTHREADS=1
    export NCCL_SOCKET_NTHREADS=1
    export NCCL_LAUNCH_MODE=GROUP
    export CUDA_VISIBLE_DEVICES=$(seq -s, "$(( NODE_RANK * GPUS_PER_NODE + DEVICE_OFFSET ))" "$(( NODE_RANK * GPUS_PER_NODE + GPUS_PER_NODE + DEVICE_OFFSET - 1 ))") 

    $SIMULATE python $CMD --dist_info "${HOST}:${BASE_PORT},${NODE_RANK},${NUM_NODES}" $@
    ) &
    echo $CUDA_VISIBLE_DEVICES
done


export NODE_RANK=0
echo CUDA_VISIBLE_DEVICES\=$(seq -s, "$(( NODE_RANK * GPUS_PER_NODE + DEVICE_OFFSET ))" "$(( NODE_RANK * GPUS_PER_NODE + GPUS_PER_NODE + DEVICE_OFFSET - 1 ))") \
      $SIMULATE python $RANK_ZERO_ADAPT_PRE $CMD $RANK_ZERO_ADAPT --dist_info "${HOST}:${BASE_PORT},${NODE_RANK},${NUM_NODES}" $@

CUDA_VISIBLE_DEVICES=$(seq -s, "$(( NODE_RANK * GPUS_PER_NODE + DEVICE_OFFSET ))" "$(( NODE_RANK * GPUS_PER_NODE + GPUS_PER_NODE + DEVICE_OFFSET - 1 ))") \
      $SIMULATE python $RANK_ZERO_ADAPT_PRE $CMD $RANK_ZERO_ADAPT --dist_info "${HOST}:${BASE_PORT},${NODE_RANK},${NUM_NODES}" $@



wait
echo "All processes finished."
