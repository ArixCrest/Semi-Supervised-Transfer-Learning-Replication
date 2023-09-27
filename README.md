# Adaptive Consistency Regularization for Semi-Supervised Transfer Learning Replication

This project includes the following folders:

1. **code**: Contains all the code.(forked and modified from [here](https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning/tree/main))
2. **Compiled Results**: Contains the test accuracies for the model trained on CIFAR and SVHN dataset.
3. **logs**: Contains all the logs generated during the training.
4. **plots**: Contains the plots for the datapoints gathered in compiled results.

## Running the Code

To run the code, follow these steps:

### Datasets
1. You can download the CIFAR-10 dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)
2. You can download the SVHN dataset from these links: [train](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) [test](http://ufldl.stanford.edu/housenumbers/test_32x32.mat)
3. CUB-200-2011 dataset can be downloaded from [here](https://data.caltech.edu/records/20098). You can preprocess the data using the `split.py` provided in the code folder.  
4. After downloading the data place the downloaded files in a **data directory** of your choice.

### Downloading the pretrained model
Pretrain the ResNet-50 model on [Imagenet](http://image-net.org/download-images) or download the Imagenet pretrained models from [pytorch.models](https://download.pytorch.org/models/resnet50-19c8e357.pth) in your checkpoint folder, e.g. `./ckpt/`. Then rename the pretrained checkpoint as `resnet_50_1.pth`.
This can be used for CUB-200-2011 datset.(Note: Training on this dataset requires around 27Gb of GPU memory.)

### Training the model
To train the model, run main.py given in `code` folder.
Example of CIFAR-10 using AKC and ARC as both 0.5.(`lambda_kd` represents AKC weight and `lambda_mmd` represents the ARC weight)

`$data_root` is the path to your downloaded data.

`$dataset` is the dataset you want to use.(cifar10 or svhn)

`$num_labels` is the number of labelled data you want for training

`$pretrain_path` is the path of the directory where pretrained models are downloaded.

   ```bash
   python -u main.py --data_root $data_root --dataset $dataset --num_labels $num_labels --pretrained_weight_path $pretrain_path --model wideresnetleaky --depth 28 --widen_factor 2 --l_batch_size 64 --ul_batch_size 448 --lambda_mmd 0.5 --lambda_kd 0.5 --lr 0.0001 --weight_decay 0.0005 --epochs 10  --coef 1.0 --alg pl --strong_aug true --threshold 0.95 --ema_teacher true  --ema_teacher_factor 0.999 --bn_momentum 0.1  --interleave 0  --seed 10 
```
## References
1. [Semi-SuperVised Transfer Learning](https://github.com/SHI-Labs/Semi-Supervised-Transfer-Learning/tree/main)
