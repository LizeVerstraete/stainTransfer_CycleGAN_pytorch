set -ex

START=$(date +%s.%N)

python train.py --dataroot {path_to_trainA_and_trainB} --name {name_of_experiment} --results_dir {path_to_results} --load_size {load_size} --crop_size {crop_size} --pool_size {pool_size} --model cycle_gan

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
