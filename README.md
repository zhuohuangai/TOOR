# TOOR
Code for "They are Not Completely Useless: Towards Recycling Transferable Unlabeled Data for Class-Mismatched Semi-Supervised Learning"

## Training
For running TOOR:
```
python3 toor.py 

optional arguments:
  -a, --alg             Choose the algorithm for running the experiment
  -r, --root            Directory root for loading the dataset
  -d, --dataset         Choose the dataset
  --num-classes         Choose the number of classes for open-set SSL
  --validation          Set the number of validate data
  --n_labels            Vary the number of labeled data
  --n_unlabels          Vary the number of unlabeled data
  -th                   Threshold to tune for detecting OOD data
```

For running SSL baseline methods:
```
python3 train_ssl.py --args_for_your_setting
```

## Note for running TOOR
To make sure the training process do not fall into overfitting, the number of training epochs shall be carefully selected for different datasets.

To prevent such an overfitting phenomenon during training, there are other tricks could be taken into consideration, such as feature mixup which is proposed in [NeurIPS 2021 Universal Semi-Supervised Learning](https://github.com/zhuohuangai/cafa-1).


## Citations
If you find our work helpful, please consider citing our paper, thank you so much!
```
@article{huang2022they,
  title={They are not completely useless: Towards recycling transferable unlabeled data for class-mismatched semi-supervised learning},
  author={Huang, Zhuo and Yang, Jian and Gong, Chen},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```

For any further problem, please feel free to contact [zhuohuang.ai@gmail.com](zhuohuang.ai@gmail.com).
