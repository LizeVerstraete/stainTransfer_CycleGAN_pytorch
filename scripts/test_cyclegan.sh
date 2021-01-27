set -ex
python train.py --dataroot ./datasets/ --name tmp --display_id -1
python test.py --dataroot  --name maps_cyclegan --model cycle_gan --phase test --no_dropout
