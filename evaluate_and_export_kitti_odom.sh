SCENARIO_NAME='20241002_171207_kitti_07-10_DEEPCLRTF'
echo "Start evaluation"
#./scripts/evaluation.py /home/usrg/scenarios --scenario kitti_07-10

echo "Finish evaluation. Start Kitti poses export ..."
./scripts/export_kitti_poses.py /deepclr/scenario/$SCENARIO_NAME

echo "Starting kitti devkit ..."
python -m kitti_devkit ${KITTI_PATH}/original/poses /deepclr/scenario/$SCENARIO_NAME/kitti
