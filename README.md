# Stain Transformation using CycleGAN in PyTorch
![exampleStainTransform](imgs/example_image_readme.png)

[comment]: <> (**CycleGAN: [Project]&#40;https://junyanz.github.io/CycleGAN/&#41; |  [Paper]&#40;https://arxiv.org/pdf/1703.10593.pdf&#41; |  [Torch]&#40;https://github.com/junyanz/CycleGAN&#41; |)

[comment]: <> ([Tensorflow Core Tutorial]&#40;https://www.tensorflow.org/tutorials/generative/cyclegan&#41; | [PyTorch Colab]&#40;https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb&#41;**)

[comment]: <> (<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>)

[comment]: <> (If you use this code for your research, please cite:)

[comment]: <> (Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.<br>)

[comment]: <> ([Jun-Yan Zhu]&#40;https://www.cs.cmu.edu/~junyanz/&#41;\*,  [Taesung Park]&#40;https://taesung.me/&#41;\*, [Phillip Isola]&#40;https://people.eecs.berkeley.edu/~isola/&#41;, [Alexei A. Efros]&#40;https://people.eecs.berkeley.edu/~efros&#41;. In ICCV 2017. &#40;* equal contributions&#41; [[Bibtex]]&#40;https://junyanz.github.io/CycleGAN/CycleGAN.txt&#41;)


[comment]: <> (Image-to-Image Translation with Conditional Adversarial Networks.<br>)

[comment]: <> ([Phillip Isola]&#40;https://people.eecs.berkeley.edu/~isola&#41;, [Jun-Yan Zhu]&#40;https://www.cs.cmu.edu/~junyanz/&#41;, [Tinghui Zhou]&#40;https://people.eecs.berkeley.edu/~tinghuiz&#41;, [Alexei A. Efros]&#40;https://people.eecs.berkeley.edu/~efros&#41;. In CVPR 2017. [[Bibtex]]&#40;https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib&#41;)

## Prerequisites
- Linux, Windows or Mac OS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
    ```bash
    git clone https://github.com/m4ln/stainNormalization_CycleGAN_pytorch.git
    cd stainNormalization_CycleGAN_pytorch
    ```
- Some sample scripts are provided in `./scripts`. Before you can run them the first time, set execute permission via
`chmod +x -R ./scripts/`

- Install dependencies via [pip](https://pypi.org/project/pip/)  
`#!./scripts/install_deps.sh`
  - [PyTorch](https://pytorch.org/get-started/locally/)
  - [visdom](https://github.com/facebookresearch/visdom)
  - For all other dependencies, use the command  
    `pip install -r requirements.txt`.
    
## Prepare Your Dataset
### CycleGAN (unpaired datasets)
To train a model on your own dataset, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. You can test your model on your training set by setting `--phase train` in `test.py`. You can also create subdirectories `testA` and `testB` if you have test data.

###Pix2Pix (paired datasets)
We provide a python script to generate pix2pix training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.

### CycleGAN train/test
- Prepare your dataset as described above
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
  ```bash
  #!./scripts/train_cyclegan.sh
  python train.py --dataroot /{path_to_train_data} --name {experiment_name}
  ```
  - To specify more train options, see the files in the directory `./options/train_options.py`
  - To see more intermediate results, check out `./results/{experiment_name}/web/index.html`.
- Test the model:
  ```bash
  #!./scripts/test_cyclegan.sh
  python test.py --dataroot /{path_to_test_data} --name {experiment_name} --model cycle_gan
  ```
  - To specify more test options, see the files in the directory `./options/test_options.py`
  - The test results will be saved to a html file here: `./results/{experiment_name}/test_latest/index.html`.
  - The (pre-)trained model is saved at `./results/{experiment_name}/latest_net_G_{AorB}.pth`.
  - The option `--model test` is used for generating results of CycleGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model cycle_gan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

### pix2pix train/test

- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- To train and test pix2pix-based colorization models, please add `--model colorization` and `--dataset_mode colorization`. See our training [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization) for more details.
- For pix2pix and your own models, you need to explicitly specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model. See this [FAQ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#runtimeerror-errors-in-loading-state_dict-812-671461-296) for more details.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## [Frequently Asked Questions](docs/qa.md)
Before you post a new question, please first look at the above Q & A and existing GitHub issues.

## Custom Model and Dataset
If you plan to implement custom models and dataset for your new applications, we provide a dataset [template](data/template_dataset.py) and a model [template](models/template_model.py) as a starting point.

## [Code structure](docs/overview.md)
To help users better understand and use our code, we briefly overview the functionality and implementation of each package and each module.

## Pull Request
You are always welcome to contribute to this repository by sending a [pull request](https://help.github.com/articles/about-pull-requests/).
Please run `flake8 --ignore E501 .` and `python ./scripts/test_before_push.py` before you commit the code. Please also update the code structure [overview](docs/overview.md) accordingly if you add or remove files.

[comment]: <> (## Citation)

[comment]: <> (If you use this code for your research, please cite our paper.)

[comment]: <> (```)

[comment]: <> (@inproceedings{StainNormalizationCycleGAN,)

[comment]: <> (  title={Normalization of HE-Stained Histological Images using Cycle-Consistent Generative Adversarial Networks},)

[comment]: <> (  author={},)

[comment]: <> (  booktitle={},)

[comment]: <> (  year={})

[comment]: <> (})

[comment]: <> (```)

### Network architecture For Cycle A to B
![scheme_AtoB](imgs/cycleGAN_scheme_AtoB.png)

### ToDo
- [ ] Add publication citation

## Acknowledgments
The project is forked from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Only a few parts are modified or added.