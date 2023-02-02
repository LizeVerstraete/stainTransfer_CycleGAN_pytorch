set -ex
# camelyon
#python train.py --dataroot /home/mr38/sds_hd/sd18a006/marlen/project_data/pytorch-cycleGan-stain-normalization/camelyon16classification/trainImages/train/ --name camelyon_onlyH_s256_c128 --max_dataset_size 5000 --preprocess resize_and_crop --load_size 256 --crop_size 128 --n_epochs 30 --n_epochs_decay 30 --lr 0.0002 --pool_size 50 --results_dir /home/mr38/sds_hd/sd18a006/marlen/project_data/pytorch-cycleGan-stain-normalization/camelyon16classification/results/
# tumorLymphnode
#python train.py --dataroot /home/mr38/sds_hd/sd18a006/marlen/project_data/pytorch-cycleGan-stain-normalization/camelyon16classification/trainImages/train/ --name tumorLymphnode_onlyH_s165_c128 --max_dataset_size 5000 --preprocess resize_and_crop --load_size 165 --crop_size 128 --n_epochs 30 --n_epochs_decay 30 --lr 0.0002 --pool_size 50 --results_dir /home/mr38/sds_hd/sd18a006/marlen/project_data/pytorch-cycleGan-stain-normalization/camelyon16classification/results/
# Graz Kollektiv
#python train.py --dataroot /home/mr38/sds_hd/sd18a006/marlen/students/erdi_semiSupervisedSegmentation/datasets/stainNormalisation/train/ --gpu_ids 0,1,2 --batch_size 32 --name normalized_to_HEV_s_c128 --max_dataset_size 5000 --preprocess resize_and_crop --load_size 512 --crop_size 128 --n_epochs 30 --n_epochs_decay 30 --lr 0.0002 --results_dir /home/mr38/sds_hd/sd18a006/marlen/students/erdi_semiSupervisedSegmentation/datasets/stainNormalisation/results/
# HEV
python train.py --dataroot /home/mr38/sds_hd/sd18a006/marlen/histoNorm/HEV/trainImages/train_test_sets/size1024_overlap341/ --gpu_ids 0,1,2 --batch_size 32  --name normalHE_to_onlyE_s1024_c64 --preprocess resize_and_crop --load_size 1024 --crop_size 64 --n_epochs 20 --n_epochs_decay 20 --lr 0.0002 --results_dir /home/mr38/sds_hd/sd18a006/marlen/histoNorm/HEV/trainImages/train_test_sets/size1024_overlap341/results/
