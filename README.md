# PSCapsNet
A PyTorch implementation of Parameter-sharing Capsule Network based on the paper [Evaluating Generalization Ability of Convolutional Neural Networks and Capsule Networks for Image Classification via Top-2 Classification](https://arxiv.org/abs/1901.10112).

## Requirements
* [Anaconda](https://www.anaconda.com/download/)
* PyTorch
```
conda install pytorch torchvision -c pytorch
```
* PyTorchNet
```
pip install git+https://github.com/pytorch/tnt.git@master
```
* capsule-layer
```
pip install git+https://github.com/leftthomas/CapsuleLayer.git@master
```

## Datasets
We have uploaded the datasets into [BaiduYun](https://pan.baidu.com/s/1wryLfOmzDnvtm28jmJhyLg)(access code:tr4u) and 
[GoogleDrive](https://drive.google.com/open?id=12LBRtmvGcJtDRqyvpk_vN-jBs0ccu1bM). 

You needn't download the datasets by yourself, the code will download them automatically.
If you encounter network issues, you can download all the datasets from the aforementioned cloud storage webs, 
and extract them into `data` directory.

## Usage

### Train model
```
python -m visdom.server -logging_level WARNING & python main.py --data_type CIFAR10 --net_mode CNN --num_epochs 300
optional arguments:
--data_type                dataset type [default value is 'MNIST'](choices:['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'])
--net_mode                 network mode [default value is 'Capsule'](choices:['Capsule', 'CNN'])
--capsule_type             capsule network type [default value is 'ps'](choices:['ps', 'fc'])
--routing_type             routing type [default value is 'k_means'](choices:['k_means', 'dynamic'])
--num_iterations           routing iterations number [default value is 3]
--batch_size               train batch size [default value is 64]
--num_epochs               train epochs number [default value is 100]
--use_da                   use data augmentation or not [default value is False]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, 
`$data_type` means the dataset type which you are training. If you want to interrupt 
this process, just type `ps aux | grep visdom` to find the `PID`, then `kill PID`.

### Visualization
```
python vis.py --data_type CIFAR10
optional arguments:
--data_type                dataset type [default value is 'STL10'](choices:['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'])
--data_mode                visualized data mode [default value is 'test_single'](choices:['test_single', 'test_multi'])
--num_iterations           routing iterations number [default value is 3]
```
Generated results are on the same directory with `README.md`.

### Generate figures
```
python utils.py
```
Generated figures are on the same directory with `README.md`.

## Results
The train loss、accuracy, test loss、accuracy and confusion matrix are showed on visdom. The pretrained models and more results can be found in [BaiduYun](https://pan.baidu.com/s/1uG8VdgNuylrWLQbKGzQ3Jg)(access code:kvmj) and 
[GoogleDrive](https://drive.google.com/drive/folders/12LBRtmvGcJtDRqyvpk_vN-jBs0ccu1bM?usp=sharing).

### MNIST
- PS-Capsule

![result](results/MNIST_Capsule_ps.png)

- FC-Capsule

![result](results/MNIST_Capsule_fc.png)

- CNN

![result](results/MNIST_CNN.png)

### FashionMNIST
- PS-Capsule

![result](results/FashionMNIST_Capsule_ps.png)

- FC-Capsule

![result](results/FashionMNIST_Capsule_fc.png)

- CNN

![result](results/FashionMNIST_CNN.png)

### SVHN
- PS-Capsule

![result](results/SVHN_Capsule_ps.png)

- FC-Capsule

![result](results/SVHN_Capsule_fc.png)

- CNN

![result](results/SVHN_CNN.png)

### CIFAR10
- PS-Capsule

![result](results/CIFAR10_Capsule_ps.png)

- FC-Capsule

![result](results/CIFAR10_Capsule_fc.png)

- CNN

![result](results/CIFAR10_CNN.png)

### STL10
- PS-Capsule

![result](results/STL10_Capsule_ps.png)

- FC-Capsule

![result](results/STL10_Capsule_fc.png)

- CNN

![result](results/STL10_CNN.png)
