# ./scripts/training.py ./configs/training/my_00-06.yaml
# ./scripts/training.py ./configs/training/dlopvt_00-06.yaml
./scripts/training.py ./configs/training/deepclr_tf_00-10.yaml
# ./scripts/training.py ./configs/training/implicit_00-06.yaml
# ./scripts/training.py ./configs/training/deepclr_tf_indy_00.yaml


# if you want to start from checkpoints.
# ./scripts/training.py ./configs/training/dlopvt_00-06.yaml --ckpt models/20230124_130203_dlopvt_00-06

# ./scripts/training.py ./configs/training/deepclr_tf_00-10.yaml --ckpt models/20230715_023325_deepclr_tf_00-06/checkpoint.tar
# ./scripts/training.py ./configs/training/deepclr_tf_00-10.yaml --ckpt models/20230712_104454_deepclr_tf_00-06/ckpt_408000.tar