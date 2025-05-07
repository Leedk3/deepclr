SCENARIO_NAME='20250507_105023_kitti_07-10_DEEPCLRTF'
echo "Start evaluation"
#./scripts/evaluation.py /home/usrg/scenarios --scenario kitti_07-10

echo "Finish evaluation. Start Kitti poses export ..."
./scripts/export_kitti_poses.py /deepclr/scenario/$SCENARIO_NAME

echo "Starting kitti devkit ..."
python -m kitti_devkit ${KITTI_PATH}/original/poses /deepclr/scenario/$SCENARIO_NAME/kitti
