# DLTemplate
A PyTorch Deep Learning Template

## Requirements
* [Anaconda(Python 3.6 version)](https://www.anaconda.com/download/)
* PyTorch(version >= 0.3.1) 
```
conda install pytorch torchvision -c pytorch
```
* PyTorchNet(version >= 0.0.1)
```
pip install git+https://github.com/pytorch/tnt.git@master
```
* tqdm(version >= 4.19.5)
```
conda install tqdm
```

## Usage
```
python -m visdom.server -logging_level WARNING & python main.py --use_da --num_epochs 300
optional arguments:
--use_da                      use data augmentation or not [default value is False]
--batch_size                  train batch size [default value is 100]
--num_epochs                  train epochs number [default value is 100]
```
Visdom now can be accessed by going to `127.0.0.1:8097/env/main` in your browser.