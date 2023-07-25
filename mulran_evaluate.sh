SCENARIO_NAME='20230725_113125_mulran_kaist_ICP_PO2PL'
echo "Start evaluation"
#./scripts/evaluation.py /home/usrg/scenarios --scenario kitti_07-10

echo "Finish evaluation. Start Kitti poses export ..."
./scripts/export_mulran2kitti_poses.py /home/usrg/scenarios/$SCENARIO_NAME

echo "Starting kitti devkit ..."
# python -m kitti_devkit ${MULRAN_PATH}/original/poses ~/scenarios/$SCENARIO_NAME/mulran
