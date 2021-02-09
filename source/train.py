# train.py
import argparse
import os

import joblib

import pandas as pd
from sklearn import metrics

import config
import model_dispatcher
import scaler_dispatcher


def run(kfold,model,scaler=None):
	#r ead the training data with folds
	df = pd.read_csv(config.TRAINING_FILE)

	# training data is where kfold is not equal to provided fold
	# reset the index
	df_train = df[df.kfold != kfold].reset_index(drop=True)

	# validation data is where kfold is equal to provided fold
	df_valid = df[df.kfold == kfold].reset_index(drop=True)

	# drop the label column from dataframe and convert it to
	# a numpy array by using .values. 
	# target is target column in the dataframe 

	x_train = df_train.drop(['target','kfold'],axis=1).values
	y_train = df_train.target.values

	#print(f'Shape of x_train: {x_train.shape}')

	#similarly, for validation, we have 

	x_valid = df_valid.drop(['target','kfold'],axis=1).values
	y_valid = df_valid.target.values

	# Scale the data using specified scaler
	# Only if the model is logistic and scaler info is provided

	if (scaler is not None) and model == 'logistic':
		sclr = scaler_dispatcher.scalers[scaler] 
		x_train = sclr.fit_transform(x_train)
		x_valid = sclr.transform(x_valid)
	else:
		pass


	# fetch the model from model_dispatcher 
	clf = model_dispatcher.models[model]
	
	# fit the model on training data
	clf.fit(x_train, y_train)

	# create predictions for validation samples
	preds = clf.predict(x_valid)
	
	# calculate & print accuracy
	F1_Score_Micro = metrics.f1_score(y_true=y_valid, y_pred=preds, average='micro')
	print(f"Fold={kfold}, Model={model}, F1_Score_Micro={F1_Score_Micro}")
	
	# save the model
	joblib.dump(clf,os.path.join(config.MODEL_OUTPUT, f"dt_{kfold}_{model}.bin"))



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--kfold",
		type=int
	)
	parser.add_argument(
		"--model",
		type=str
	)
	parser.add_argument(
		"--scaler",
		type=str
	)

	args = parser.parse_args()
	if args.scaler:

		run(
			kfold=args.kfold,
			model=args.model,
			scaler=args.scaler
		)
	else:
		run(
			kfold=args.kfold,
			model=args.model
		)

