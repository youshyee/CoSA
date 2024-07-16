#!/bin/bash
set -x
NAME=EXP_VOC
EXP_DIR=$HOME/CoSA_EXP
RANDPORT=6531
echo Using port $RANDPORT
torchrun --master_port $RANDPORT --nproc_per_node=2 main.py $NAME --work_dir $EXP_DIR \
	--dataset VOC12 \
	--voc12_root $HOME/data/VOCdevkit/VOC2012 \
	--max_iters 32000 \
	--aux_layer -4 \
