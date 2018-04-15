# SPCapsNet
A PyTorch implementation of Shared Parameters Capsule Network based on NIPS2018 paper [Share parameters between capsules with k-means routing for image classification]()

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
* tqdm
```
conda install tqdm
```

## Usage
```
python -m visdom.server -logging_level WARNING & python main.py --data_type CIFAR10 --net_mode CNN --num_epochs 300
optional arguments:
--data_type                   dataset type [default value is 'MNIST'](choices:['MNIST', 'FashionMNIST', 'SVHN', 'CIFAR10', 'STL10'])
--net_mode                    network mode [default value is 'Capsule'](choices:['Capsule', 'CNN'])
--use_da                      use data augmentation or not [default value is False]
--routing_type                routing type [default value is 'k_means'](choices:['k_means', 'dynamic'])
--num_iterations              routing iterations number [default value is 3]
--batch_size                  train batch size [default value is 50]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/$data_type` in your browser, 
`$data_type` means the dataset type which you are training.

## Results
The train loss、accuracy, test loss、accuracy, and confusion matrix are showed with visdom.

### MNIST
- Capsule

![result](results/capsule_mnist.png)

- CNN

![result](results/cnn_mnist.png)

### FashionMNIST
- Capsule

![result](results/capsule_fashionmnist.png)

- CNN

![result](results/cnn_fashionmnist.png)

### SVHN
- Capsule

![result](results/capsule_svhn.png)

- CNN

![result](results/cnn_svhn.png)

### CIFAR10
- Capsule

![result](results/capsule_cifar10.png)

- CNN

![result](results/cnn_cifar10.png)

### STL10
- Capsule

![result](results/capsule_stl10.png)

- CNN

![result](results/cnn_stl10.png)