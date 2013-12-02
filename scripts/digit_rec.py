#This script uses a support vector machine to recognize digits.

import numpy as np
import pandas as pd
from random import sample
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm

train_data = "digits\\train.csv"
test_data = "digits\\test.csv"

#create dataset
def load_clean(filename):
	training = pd.read_csv(filename)
	Data = training.values
	
	return Data
	
	
def create_index(data, n):
	'''Helper function to be used to help randomly sample data
	'''
	train_ix = sample(xrange(len(data)) , n)
	test_ix = [i for i in xrange(2*n) if i not in train_ix]
	return train_ix, test_ix
	
	
def create_training_data(data, num_samples =3000):
	"""
	Creates training and test sets by randomly shuffling the data
	"""
	rindex_train, rindex_test = create_index(data, num_samples)
	X_train = data[rindex_train,1:]
	y_train = data[rindex_train,0]
	X_test = data[rindex_test,1:]
	y_test = data[rindex_test,0]
	return X_train, y_train, X_test, y_test


#implememtsuppor vector machine to model the digits on 1000 samples

def predictions(model, x_train,y_train, x_test):
	clf = model
	clf.fit(x_train,y_train)
	y_pred = clf.predict(x_test)
	return y_pred, clf

#calculate the accuracy

def model_acc(predictions, validation):
	positives = 0
	for label in xrange(len(predictions)):
		if predictions[label] == validation[label]:
			positives +=1
	return 100.0 * positives/len(predictions)


def recognize(model, filename):
	''' Apply the model to new data
	'''
	data = load_clean(filename)
	X_tr, y_tr, X_ts, y_ts = create_training_data(data)
	y_prd, clf = predictions(model,X_tr, y_tr, X_ts)
	accuracy = model_acc(y_prd, y_ts)
	return accuracy, clf
	
def submit_probabilities(model, train_file, test_file):
	''' Create probabilities for the submission
	'''
	acc, clf = recognize(model, train_file)
	validate_data = load_clean(test_file)
	for m in xrange(len(validate_data)):
		print clf.predict(validate_data[m, :])[0]
	
if __name__=="__main__":
	print recognize(svm.SVC(kernel='polynomial'),train_data)
	
	
	