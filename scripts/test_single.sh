set -ex
START=$(date +%s.%N)
python test.py --dataroot ./data --name short --model test --direction AtoB --model_suffix "_A" --dataset_mode single --no_dropout
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)