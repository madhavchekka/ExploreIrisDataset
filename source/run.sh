#!/bin/sh

python train.py --kfold 0 --model rf
python train.py --kfold 1 --model rf
python train.py --kfold 2 --model rf
python train.py --kfold 3 --model rf
python train.py --kfold 4 --model rf


python train.py --kfold 0 --model decision_tree_gini
python train.py --kfold 1 --model decision_tree_gini
python train.py --kfold 2 --model decision_tree_gini
python train.py --kfold 3 --model decision_tree_gini
python train.py --kfold 4 --model decision_tree_gini


python train.py --kfold 0 --model decision_tree_entropy
python train.py --kfold 1 --model decision_tree_entropy
python train.py --kfold 2 --model decision_tree_entropy
python train.py --kfold 3 --model decision_tree_entropy
python train.py --kfold 4 --model decision_tree_entropy


python train.py --kfold 0 --model logistic --scaler standard
python train.py --kfold 1 --model logistic --scaler standard
python train.py --kfold 2 --model logistic --scaler standard
python train.py --kfold 3 --model logistic --scaler standard
python train.py --kfold 4 --model logistic --scaler standard


python train.py --kfold 0 --model logistic --scaler minmax
python train.py --kfold 1 --model logistic --scaler minmax
python train.py --kfold 2 --model logistic --scaler minmax
python train.py --kfold 3 --model logistic --scaler minmax
python train.py --kfold 4 --model logistic --scaler minmax


python train.py --kfold 0 --model logistic --scaler maxabs
python train.py --kfold 1 --model logistic --scaler maxabs
python train.py --kfold 2 --model logistic --scaler maxabs
python train.py --kfold 3 --model logistic --scaler maxabs
python train.py --kfold 4 --model logistic --scaler maxabs

