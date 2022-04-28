set -ex
# set '--model_suffix _A' to test the model from A to B, or '_B' vice versa

# normalize from Camelyon16 to HEV stain only
#python test.py --dataroot /home/mr38/sds_hd/sd18a006/marlen/project_data/pytorch-cycleGan-stain-normalization/camelyon16classification/trainImages/test --name camelyon_onlyH_s256_c128 --model test --phase test --no_dropout --preprocess none --epoch 60 --suffix camelyon --num_test 10000 --model_suffix _A --results_dir /home/mr38/sds_hd/sd18a006/marlen/project_data/pytorch-cycleGan-stain-normalization/camelyon16classification/results/
# HEV
python test.py --dataroot /home/mr38/sds_hd/sd18a006/marlen/histoNorm/HEV/trainImages/train_test_sets/size1024_overlap341/test --suffix normalHE --name normalHE_to_onlyE_s1024_c64 --model test --phase test --no_dropout --preprocess none --epoch latest --num_test 100 --model_suffix _A --results_dir /home/mr38/sds_hd/sd18a006/marlen/histoNorm/HEV/trainImages/train_test_sets/size1024_overlap341/results/
