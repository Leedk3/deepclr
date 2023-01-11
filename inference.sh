export MODEL_PATH=/home/usrg/deepclr/models
MODEL=20230110_213507_dlopvt_00-06
WEIGHT=weights.tar #default : weights.tar

# kitti 07-10 validation set
# ./scripts/inference.py ./configs/scenarios/kitti_07-10.yaml $MODEL ~/scenarios --weights $WEIGHT

# kitti 04 & 10
./scripts/inference.py ./configs/scenarios/kitti_07-10.yaml $MODEL ~/scenarios --weights $WEIGHT

# kitti 01-10 training set
# ./scripts/inference.py ./configs/scenarios/kitti_01-10.yaml $MODEL ~/scenarios --weights $WEIGHT
