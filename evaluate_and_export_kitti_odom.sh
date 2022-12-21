SCENARIO_NAME='20221221_134844_kitti_07-10_PVNAVI'
echo "Start evaluation"
./scripts/evaluation.py /home/usrg/scenarios --scenario kitti_07-10

echo "Finish evaluation. Start Kitti poses export ..."
./scripts/export_kitti_poses.py /home/usrg/scenarios/$SCENARIO_NAME

echo "Starting kitti devkit ..."
python -m kitti_devkit ${KITTI_PATH}/original/poses ~/scenarios/$SCENARIO_NAME/kitti
