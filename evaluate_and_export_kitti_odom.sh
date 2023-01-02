SCENARIO_NAME='test/20221228_192213_kitti_04_10_PVNAVI'
echo "Start evaluation"
#./scripts/evaluation.py /home/usrg/scenarios --scenario kitti_07-10

echo "Finish evaluation. Start Kitti poses export ..."
./scripts/export_kitti_poses.py /home/usrg/scenarios/$SCENARIO_NAME

echo "Starting kitti devkit ..."
python -m kitti_devkit ${KITTI_PATH}/original/poses ~/scenarios/$SCENARIO_NAME/kitti
