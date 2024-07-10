#!/bin/bash
set -x
NAME=EXP_COCO
EXP_DIR=$HOME/CoSA_EXP
RANDPORT=6541
echo Using port $RANDPORT
torchrun --master_port $RANDPORT --nproc_per_node=2 main.py $NAME --work_dir $EXP_DIR \
	--dataset COCO \
	--coco_root $HOME/data/coco/
