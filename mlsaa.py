# -*- coding: utf-8 -*-
"""Master Thesis Work

Copyright (c) 2018, Yassin Haj Ahmad
All rights reserved.

The master thesis work has been carried out at the School of Engineering in Jönköping in
	the subject area “Sentiment Analysis using Multi-label Machine Learning Techniques”.
	The work is a part of the two-years Master of Science programme, Software Product 
	Engineering. The author takes full responsibility for opinions, conclusions and
	findings presented.

Papar title:
	Mining Comparative Opinion using Multi-label Machine Learning Techniques

Paper sub-title:
	A case study to identify comparative opinions, based on product aspects, and their
	sentiment classification, in online customer reviews..

Paper within:
	Software Product Engineering Master’s Program

Scope: 
	30 credits (second cycle)

Author: 
	Yassin Haj Ahmad

Supervisor: 
	Johannes Schmidt

Examinar:
	Ulf Johansson

In:
	JÖNKÖPING December 2018
"""
# About this module
"""
Module name:
	The name of this module is: "MLSAA: Multi-label Sentiment Analysis Application"

This module demonstrates the software developed for the multi-label sentiment
	analysis application used in the master thesis work. The module  contains the 
	required functions by the application. It has the functions to prepare the 
	source and development datasets. Also, the functions to preprocess the 
	labeled dataset, and run the experiments on the classification model. 

The main functions discussed in the thesis are listed below:
	read_source: Reads the reviews from the source dataset.
	prepare_development: Create the development dataset from the source.
	preprocess: Preprocess the text in the labeled dataset.
	transform_labels: Transforms the labels into a multi-label format.
	classify: Run the classification model which includes the multi-label 
	classification technique, machine learning classifier, dataset sampling,
		features selection, and evaluation functions.


Example:
		$ python
        >>> import mlsaa
        >>> mlsaa.classify(d="dataset_name", ct=1, pt=1, cl=1)
"""
# Importing general required modules
import pandas as pd
import numpy as np
import random
import pymysql
import re
import unicodedata
import warnings
import os.path
import sys
from enum import Enum
import time
import matplotlib.pyplot as plt
import itertools

# Import teh module for handling contractions
from pycontractions import Contractions

# Importing all required nltk modules
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.sentiment.util import mark_negation
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Importing all required scikit-learn modules
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import hamming_loss as hamming_loss_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# Importing all required scikit-multilearn modules
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from skmultilearn.adapt import MLkNN
from skmultilearn.ensemble import RakelD

# A module variable for the classification model
classification_model = None
"""Attributes:
    classification_model (classifier): as a module level variable to apply prediction on new data 
 		if needed using predict function after training.
"""

# The function contains the code to sample the lablede dataset after preprocessing,
# 	train and test the classfication model, and evaluating it.
def classify(dataset_name = "preprocessed_labeled", ct = 1, pt = 1, cl = 1,
	ts = 0.25, fo = 10, bi = True, mf = 5000, minn = 1, maxn = 3,
	se = True, co = True, ab = True, dt = False, cf = False):

	"""The function contains the code to sample the labeled dataset after preprocessing,
			train and test the classfication model, and evaluating it.

    Args:
        dataset_name (str): The name of the dataset to be classified.
        ct (int): The number of the classification technique as in Table 4 in the paper.
        pt (int): The number of the problem transformation method as in Table 5 in the paper.
        cl (int): The number of the classifier as in Table 6 in the paper.
        ts (int): The test dataset size (IV2 in Table 14 in the paper).
        fo (int): The number of folds for the cross validations (IV5 in Table 14 in the paper).
        bi (bool): The features as Bag of Words (BoW) with the occurances of words as binary.
        	Value is True by defaul. If set to False the count of occurances will be given 
        	(IV6 in Table 14 in the paper).
        mf (int): Maximum number of features (IV7 in Table 14 in the paper).
        minn (int): Minimum number of words in the ngram (IV8 in Table 14 in the paper).
        maxn (int): Maximum number of words in the ngram (IV9 in Table 14 in the paper).
        se (bool): The sentiment label. True by Defaul for teh multi-label problem. 
        	Can be set to False if binary classification is needed on other labels only.
        co (bool): The comparative label. True by Defaul for teh multi-label problem. 
        	Can be set to False if binary classification is needed on other labels only.
        ab (bool): The aspect-based label. True by Defaul for teh multi-label problem. 
        	Can be set to False if binary classification is needed on other labels only.
        dt (bool): The report details in each cross validation fold. Set to True if all 
        	metrics needed. It was used when doing binary classification mostly (Not tested
        	with multi-label).
		cf (bool): Prints the confusion matrix if true (Used for evaliation purpose with
			binary classification.

    Returns:
        None
    """

	# Initialize the classification model
	global classification_model

	# Get the number / count of labels to determine if it is single or multi-label problem
	labels_count = 0
	multi_label = False
	# Variables to set the labels names, classes names and classification type for reporting
	labels_names = []
	classes_names = []
	classification_type = "Binary"
	if se: 
		labels_count +=1
		labels_names.append("L1 - Sentiment")
		classes_names.append("C1 - Positive")
		classes_names.append("C2 - Negative")
	if co: 
		labels_count +=1
		labels_names.append("L2 - Comparative")
		classes_names.append("C3 - Comparative")
		classes_names.append("C4 - Non-comparative")
	if ab: 
		labels_count +=1
		labels_names.append("L3 - Aspect-based")
		classes_names.append("C5 - Aspect-based")
		classes_names.append("C6 - Non-aspect-based")
	if labels_count > 1: 
		multi_label = True
		classification_type = "Multi-label"		

	# Execut only if there is at least one label to classify, 
	# i.e. at least one of labels paramters (se, co, ab) is True
	if labels_count > 0:

		# Get the reviews text array from the preprocessed labeled dataset
		reviews = get_reviews(dataset_name=dataset_name)

		# Get labels array based on required labels to be predicted and the number of labels
		# If it is multi-label problem, the function returns a binarized labels
		# If not multi label, the fucnction returns a list of one list for one label
		labels = get_labels(dataset_name=dataset_name, sentiment=se, 
							comparative=co, aspects_based=ab)

		# A variable for the classfier name used when reporting at teh end of the function
		classifier_name = ""

		# A variable for the classfier used in the classification model, set based on cl arg
		classifier = None

		# A variable for tfidf transfomer when needed with some classifiers such as kNN
		tfidf_tranformer = None
		
		# Select classifier based on the given arguments
		if cl == 1: 
			classifier = BernoulliNB()
			classifier_name = "CL1 - NB - Naive Bayes"
		elif cl == 2: 
			classifier = LinearSVC()
			classifier_name = "CL2 - SVM - Support Vector Machine"
		elif cl == 3: 
			classifier = LogisticRegression()
			classifier_name = "CL3 - MaxEnt - Maximum Entropy"
		elif cl == 4: 
			classifier = KNeighborsClassifier(n_neighbors=100)
			classifier_name = "CL4 - kNN - k-Nearest Neighbors"
			tfidf_tranformer = TfidfTransformer()
		elif cl == 5: 
			classifier = OneVsRestClassifier(DecisionTreeClassifier(max_depth=12))
			classifier_name = "CL5 - DT - Decision Trees"
		elif cl == 6: 
			classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=300, max_depth=15))
			classifier_name = "CL6 - RF - Random Forest"
		elif cl == 7: 
			classifier = MLkNN(k=150, s=0.5)
			tfidf_tranformer = TfidfTransformer(use_idf=True)
			classifier_name = "CL7 - MLkNN - Multi-label kNN"
		else: 
			print("Please select a valid classifier number for cl.") 
			print("It can only be from 1-7 as in Table 6 in the paper.")
			return

		# Check the techniques and methods based on ct and pt args
		technique_name = ""
		method_name = ""
		if multi_label:
			if ct == 1: # Problem Transformtion technique
				technique_name = "CT1 - Problem Transformtion"
				if pt == 1: 
					classifier = BinaryRelevance(classifier) # Binary Relevance method
					method_name = "PT1 - Binary Relevance"
				elif pt == 2: 
					classifier = LabelPowerset(classifier) # Label Powerset method
					method_name = "PT2 - Label Powerset"
				elif pt == 3: 
					classifier = ClassifierChain(classifier) # Classifier Chain method
					method_name = "PT3 - Classifier Chain"
				elif not (cl == 4 or cl == 5 or cl == 6):
					method_name = "N/A"
					print("Please select a valid method number for pt with the selected classifier.")
					print ("It can only be from 1-3 as in Table 5 in the paper")
					return
			elif ct == 2:
				technique_name = "CT2 - Algorithm Adaptation"
				if not cl == 7:
					print("Classifier number 7, CL7 - MLkNN, can be used with this technique only.")
					return
			elif ct == 3: # Ensemble (RAKEL) technique
				technique_name = "CT3 - Ensemble (RAKEL)"
				classifier = RakelD(
				    base_classifier=classifier,
				    base_classifier_require_dense=[False, False],
				    labelset_size=3 # As suggested by the original paper
				)
			else:
				print("Please select a valid classification technique number for ct. It can only be from 1-3 as in Table 4 in the paper")
				return

		# The classifcation model which is a Pipeline of vectorizers and transformers for the
		# features selection technique
		classification_model = Pipeline([
			('vectorizer', CountVectorizer(analyzer="word",
		    							   ngram_range=(minn, maxn),
		    							   tokenizer=word_tokenize,
		    							   max_features=mf,
		    							   binary=bi)),
			('tfidf', tfidf_tranformer),
			('classifier', classifier)
		])

		# Print the selected properties for the classification model
		print("-------------------------------------------------------------------------")
		print("Number of labebled reviews:", len(reviews))
		print("Number of features:", mf)
		print("Labels:", ", ".join(labels_names))
		print("Classes:", ", ".join(classes_names[:3]))
		if multi_label: print("\t", ", ".join(classes_names[3:]))
		print("Classification type:", classification_type)
		if multi_label: 
			print("Classification technique:", technique_name)
			print("Classification method:", method_name)
		print("Classifier:", classifier_name)
		

		# Creat empty arrays for the metrics (scores) with length of the folds to avaraged later
		# *** This needs to optimized (Future Work)
		scores, macro_scores = ([None] * fo for _ in range(2))
		recall, precision, f1_score, support = ([0.0] * fo for _ in range(4))
		accuracy, hamming_loss, total_time = ([0.0] * fo for _ in range(3))
		# Arrays for individual scores for the classes
		p1, p2, p3, p4, p5, p6 = ([0.0] * fo for _ in range(6))
		r1, r2, r3, r4, r5, r6 = ([0.0] * fo for _ in range(6))
		f1, f2, f3, f4, f5, f6 = ([0.0] * fo for _ in range(6))
		s1, s2, s3, s4, s5, s6 = ([0.0] * fo for _ in range(6))

		# Temporary variables to check the best predictions in binary classification
		# This is used for evaluating the labeled dataset as in the paper
		best_test_y = None
		best_y_pred = None
		best_acc = 0

		# Split the dataset into training and testing using random sampling.
		# The functions takes the number of fold and generates an arrays of indicies for all folds.
		ss = ShuffleSplit(n_splits=fo, test_size=ts, random_state=28)

		# Save the cross validation itteration number for the loop (for printing and reporting purpose)
		cv = 0

		# Starts the cross validations loop
		# The loop will iterate on the array of indicies given by the sampling method above.
		# The array indicies has a total number of elements equal to the folds,
		# so loop will iterate by the number of folds.
		for train_index, test_index in ss.split(reviews, labels):
			print("---------------------------Cross validation #"+str(cv+1)+"---------------------------")
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore")

				# Splits the reviews text and labels array into training and testing
				train_X, test_X = reviews[train_index], reviews[test_index]
				train_y, test_y = labels[train_index], labels[test_index]

				# Training and testing the classification model
				# The result is an array of predicted labels for the test dataset stored in y_pred
				print("Training, testing and evaluating the classification model...")
				start = time.time()
				y_pred = classification_model.fit(train_X, train_y).predict(test_X)
				end = time.time()

				# Calculate the total time for training and testing
				total_time[cv] = end-start

				# Get the macro for precision, recall and f1-score metrics
				macro_scores = precision_recall_fscore_support(test_y, y_pred, average="macro")			
				precision[cv] = macro_scores[0]
				recall[cv] = macro_scores[1]
				f1_score[cv] = macro_scores[2]

				# Get the precision, recall, f1-score and support metrics per class per label
				# by setting average = None in the scoreing function
				if multi_label:
					# *** This needs to optimized (Future Work)
					scores = precision_recall_fscore_support(test_y, y_pred, average=None)
					p1[cv],p2[cv],p3[cv],p4[cv] = scores[0][0],scores[0][1],scores[0][2],scores[0][3]
					r1[cv],r2[cv],r3[cv],r4[cv] = scores[1][0],scores[1][1],scores[1][2],scores[1][3]
					f1[cv],f2[cv],f3[cv],f4[cv] = scores[2][0],scores[2][1],scores[2][2],scores[2][3]
					s1[cv],s2[cv],s3[cv],s4[cv] = scores[3][0],scores[3][1],scores[3][2],scores[3][3]
					if labels_count > 2:
						p5[cv],p6[cv] = scores[0][4],scores[0][5]
						r5[cv],r6[cv] = scores[1][4],scores[1][5]
						f5[cv],f6[cv] = scores[2][4],scores[2][5]
						s5[cv],s6[cv] = scores[3][4],scores[3][5]
					support[cv] = s1[cv]+s2[cv]+s3[cv]+s4[cv]+s5[cv]+s6[cv]

				# Get the accuracy metric
				accuracy[cv] = accuracy_score(test_y, y_pred)
				# Get the hamming loss metric
				hamming_loss[cv] = hamming_loss_score(test_y, y_pred)

				# Printing metrics for each cross validation
				# if dt = Tru a full report is printed in each cross validation iteration
				if dt:
					print(classification_report(test_y, y_pred, target_names=classes_names, digits=3))
				else:
					print("Macro Recall: %0.3f" % recall[cv])
					print("Macro Precision: %0.3f" % precision[cv])
					print("Macro F1-Score: %0.3f" % f1_score[cv])
				print("Hamming loss: %0.3f" % hamming_loss[cv])
				print("Accuracy: %0.3f" % accuracy[cv])
				print("Total Time: %0.3f" % total_time[cv])

				# Get teh best accuracy score for evaluation purposes using binary classification
				if accuracy[cv] > best_acc:
					best_acc = accuracy[cv]
					best_test_y = test_y
					best_y_pred = y_pred
			# Increase the cross validation for printing
			cv+=1

		# Print the average of metrics over all the folds of cross validations
		print("-------------------Total of "+str(fo)+"-folds cross validations--------------------")
		if multi_label:
			precision = np.asarray(precision).mean()
			recall = np.asarray(recall).mean()
			f1_score = np.asarray(f1_score).mean()
			support = np.asarray(support).mean()
			print("Label\tClass\tPrecision\tRecall\tF1-score\tSupport")
			print("L1\tC1\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (np.asarray(p1).mean(),np.asarray(r1).mean(),np.asarray(f1).mean(),np.asarray(s1).mean()))
			print("  \tC2\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (np.asarray(p2).mean(),np.asarray(r2).mean(),np.asarray(f2).mean(),np.asarray(s2).mean()))
			print("L2\tC3\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (np.asarray(p3).mean(),np.asarray(r3).mean(),np.asarray(f3).mean(),np.asarray(s3).mean()))
			print("  \tC4\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (np.asarray(p4).mean(),np.asarray(r4).mean(),np.asarray(f4).mean(),np.asarray(s4).mean()))
			if labels_count > 2:
				print("L3\tC5\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (np.asarray(p5).mean(),np.asarray(r5).mean(),np.asarray(f5).mean(),np.asarray(s5).mean()))
				print("  \tC6\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (np.asarray(p6).mean(),np.asarray(r6).mean(),np.asarray(f6).mean(),np.asarray(s6).mean()))
			print("Macro\t\t%0.3f\t\t%0.3f\t%0.3f\t\t%0.0f" % (precision,recall,f1_score,support))
		else:
			print(classification_report(best_test_y, best_y_pred, target_names = classes_names, digits=3))
		print("-------------------------------------------------------------------------")
		print("Macro F1-Score: %0.3f (+/- %0.3f)" % (np.asarray(f1_score).mean(), np.asarray(f1_score).std() * 2))
		print("Hamming loss: %0.3f (+/- %0.3f)" % (np.asarray(hamming_loss).mean(), np.asarray(hamming_loss).std() * 2))
		print("Accuracy: %0.3f (+/- %0.3f)" % (np.asarray(accuracy).mean(), np.asarray(accuracy).std() * 2))
		print("Total time: %0.3f (+/- %0.3f)" % (np.asarray(total_time).mean(), np.asarray(total_time).std() * 2))
		print("-------------------------------------------------------------------------")

		# If not  multi-label problem prins the confusion matrix
		if not multi_label and cf:
			cm = confusion_matrix(best_test_y, best_y_pred)
			plt.figure()
			plot_confusion_matrix(cm, classes_names)
			plt.show()
			print("-------------------------------------------------------------------------")
	else:
		print("At least one label need to be classified.")
		print("Either se, co or ab is set to True in arguments)")

# Get sets of labels for training, prediction and testing
# For multi label they are transformed for the problem using MultiLabel Binarizer function
def get_labels(dataset_name = "preprocessed_labeled", sentiment = True,
			comparative = False, aspects_based = False):

	"""The function returns the labels array as a signle-label or transformed
		into a multi-label format.
    Args:
        dataset_name (str): The name of the dataset to be classified.
        sentiment (bool): Get the sentiment label or not.
        comparative (bool): Get the comparative label or not.
        aspects_based (bool): Get the aspect-based label or not.

    Returns:
        array: either an array of single label for binary, or an array of 
        binarized multi-labels
    """
	# Initialize a list of arrays based on the labels count
	labels = []

	# Get labels count and if it is a multi-label
	multi_label = False
	labels_count = 0
	if sentiment: labels_count +=1
	if comparative: labels_count +=1
	if aspects_based: labels_count +=1
	if labels_count > 1:
		multi_label = True

	# Check if the labels count is greater than 1, otherwise, fail the function.
	if labels_count > 0:
		data = load_dataset_from_csv(dataset_name)
		dataset_size = len(data.index)

		# If multi-label get a transformed labels
		if multi_label:
			# Loop on the labels count which equals to the dataset size
			for j in range(dataset_size):
				# The labels are stored as a set of labels such as (C1, C4, C6)
				labels_set = set()
				if sentiment:
					labels_set.add("C1" if data["sentiment_polarity_label"][j] == "Positive" else "C2")
				if comparative:
					labels_set.add("C3" if data["is_comparative_label"][j] == "Yes" else "C4")
				if aspects_based:
					labels_set.add("C5" if data["is_aspects_based_label"][j] == "Yes" else "C6")
				labels.append(labels_set)

			# Label trnasformation in binary values using the function below.
			# Discussed in section 3.1.4 in the paper.
			mlb = MultiLabelBinarizer()
			labels = mlb.fit_transform(np.asarray(labels))
		
		# If not multi-label a basic numpy array is returned with one label as one column
		else:
			for i in range(dataset_size):
				label = ""
				if sentiment:
					label += "C1" if data["sentiment_polarity_label"][i] == "Positive" else "C2"
				if comparative:
					label += "C3" if data["is_comparative_label"][i] == "Yes" else "C4"
				if aspects_based:
					label += "C5" if data["is_aspects_based_label"][i] == "Yes" else "C6"
				labels.append(label)
			labels = np.asarray(labels)
	else:
		print("At least one label need to be set to True in arguments")
		return None
	return labels

# Funtion to get reviews text array for the classification model
def get_reviews(dataset_name = "preprocessed_labeled", pos_tagged = False):
	"""The function returns the reviews text array from the dataset.

    Args:
        dataset_name (str): The name of the dataset to be classified.
        pos_tagged (bool): Get a pos_tagged text of the reviews.

    Returns:
        array: the reviews text
    """
	data = load_dataset_from_csv(dataset_name)

	# Only if pos tagged, replace the reviews text with pos tags (used for subjectivity classification)
	if pos_tagged:
		review_text = ""
		data_length = len(data.index)
		for i in range(data_length):
			tagged_review = to_pos_tags(str(data["review_text"][i]))
			review_text = tagged_review.strip()
			data.loc[i,"review_text"] = review_text

			loading(i, data_length, "Getting a total of "+str(data_length)+" reviews")

	# Get the reviews text
	reviews = np.asarray(data["review_text"])
	return reviews

# The function to preprocess the labeled dataset:
# All the args that are set to False are fund to be not helpful in the classfication
# performance, so they are not used by default, but can be tested if needed
def preprocess(dataset_name="labeled", lemmatize=False, stem=False,
			subjective=False, negation=False, remove_digits=False):
	"""The function processes the labeled dataset and exports a preprocessed
			CSV file of the labeled dataset.

    Args:
        dataset_name (str): The name of the dataset to be classified.
        lemmatize (bool): Lemmatize the reviews text.
        stem (bool): Stem the reviews text.
        subjectivity (bool): Remove objective sentences and keep the 
        	subjective ones from teh reviews text.
        negation (bool): Negation handling the reviews text.
        remove_digits (bool): Removes the digists the reviews text.

    Returns:
        None
    """
	data = load_dataset_from_csv(dataset_name)
	# In addiiton to reviews text, the comparative and aspects-based sentences 
	# identified are also preprocessed.
	keys = ["review_text", "comparative_sentences", "aspects_based_sentences"]
	dataset_size = len(data.index)

	# Loop over the dataset size
	for i in range(dataset_size):
		for k in keys:
			text = str(data[k][i])
			if text and text != "nan":
				processed_text = process_text(text, lemmatize=lemmatize, stem=stem,
										subjective=subjective, negation=negation,
										remove_digits=remove_digits)
				data.loc[i,k] = processed_text
		loading(i+1, data_length, "Processing data in a total of "+str(data_length))
	
	# Exports the preprocessed dataset to the datasets folder with preprocessed_ tag
	data[["review_id", "review_key", "review_type", "review_text", "review_rating",
		"review_helpful", "sentiment_polarity_label", "is_comparative_label",
		"is_aspects_based_label", "comparative_sentences", "aspects_based_sentences",
		"review_status"]].to_csv("./datasets/preprocessed_"+dataset_name+".csv",
		sep="\t", quoting=3, index=False)

# a sub function to process one review in the function above
def process_text(text, lemmatize=False, stem=False, subjective=False, negation=False, remove_digits=False):
	processed_text = ""
	if text and text != "nan":
		processed_text = text.lower()
		processed_text = remove_white_spaces(processed_text)
		processed_text = process_symbols(processed_text)
		processed_text = remove_accented_chars(processed_text)
		processed_text = remove_special_characters(processed_text)
		if remove_digits:
			processed_text = remove_digits(processed_text)
		if subjective:
			processed_text = subjective(processed_text)
		if stem:
			processed_text = stem_text(processed_text)
		if lemmatize:
			processed_text = lemmatize_text(processed_text)
		if negation:
			processed_text = process_negation(processed_text)
			processed_text = replace_all(processed_text, ",_NEG", ",")
		processed_text = process_common_mistakes(processed_text)
		processed_text = process_symbols(processed_text)
		processed_text = remove_white_spaces(processed_text)
	return processed_text

# Removes Accented Chars: replace letters with accent notations such as “ê ä ó”
# with the normal English letters “e a o”. These characters may lead to different
# variations of a word. This can be an influencing factor when building the
# classification model. 
def remove_accented_chars(text):
	text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	return text

# Removing of unnecessary special characters: only these special characters are kept
# in the review text “. , ' - _ $ % ? !” because they may have a meaning in the sentence.
# All other special characters were removed.  
def remove_special_characters(text):
	text = remove_empty_lines(text)
	pattern = r"[^.,'\-_/$%?!a-zA-z0-9\s]"
	text = re.sub(pattern, " ", text)
	return text

# Function to remove digits (Tested only, not used in the paper)
def remove_digits(text):
	pattern = r"[0-9]"
	text = re.sub(pattern, " ", text)
	return text

# Removes empty lines (This is considered under special characters handling in the paper)
def remove_empty_lines(text):
	text = text.replace("\r\n", " ")
	text = text.replace("\n", " ")
	text = text.replace("\r", " ")
	text = text.replace("\t", " ")
	return text

# Removes white spaces (This is considered under special characters handling in the paper)
def remove_white_spaces(text):
	text = text.replace(".", ". ")
	text = text.replace(",", ", ")
	text = replace_all(text,". .", ".")
	text = replace_all(text,", ,", ",")
	text = replace_all(text,". ,", ",")
	text = replace_all(text,", .", ".")
	text = replace_all(text,"..", ".")
	text = replace_all(text,",,", ",")
	text = replace_all(text,".,", ",")
	text = replace_all(text,",.", ".")
	text = replace_all(text,"  ", " ")
	text = replace_all(text," , ", ", ")
	text = replace_all(text," . ", ". ")
	text = replace_all(text,"..", ".")
	text = replace_all(text,",,", ",")
	text = replace_all(text,".,", ",")
	text = replace_all(text,",.", ".")
	text = replace_all(text,". .", ".")
	text = replace_all(text,", ,", ",")
	text = replace_all(text,". ,", ",")
	text = replace_all(text,", .", ".")
	text = replace_all(text," , ", ", ")
	text = replace_all(text," . ", ". ")
	text = replace_all(text,"  ", " ")
	return text

# Removes some special symbols and replace them with more suitable characters for the analysis
# This is considered under special characters handling in the paper
def process_symbols(text):
	text = process_symbol(text, "$", "money")
	text = process_symbol(text, "%", "percent")
	text = process_symbol(text, '"', ",")
	text = process_symbol(text, '(', ",")
	text = process_symbol(text, ')', ",")
	text = process_symbol(text, '[', ",")
	text = process_symbol(text, ']', ",")
	text = process_symbol(text, '{', ",")
	text = process_symbol(text, '}', ",")
	text = process_symbol(text, '<', ",")
	text = process_symbol(text, '>', ",")
	return text

# Help function to process symbols above
def process_symbol(text, symbol, replacement):
	text = replace_all(text,symbol+" ", symbol)
	text = replace_all(text," "+symbol, symbol)
	text = replace_all(text,symbol+symbol, symbol)
	text = text.replace(symbol, " "+replacement+" ")
	return text

# Removes white spaces (This is minor observation in some rebews, not discussed in the paper)
def process_common_mistakes(text):
	text = text.replace("alot", "a lot")
	return text

# Negation handling (Tested only, not used in the paper)
def process_negation(text):
	negation_text = mark_negation(word_tokenize(text, ))
	text = " ".join(negation_text)
	return text

# Lemmatize text (Tested only, not used in the paper)
def lemmatize_text(text):
	lemmatizer = WordNetLemmatizer()
	sentences = sent_tokenize(text)
	for sentence in sentences:
		tagged_sentence = pos_tag(word_tokenize(sentence))
		for word, tag in tagged_sentence:
			wn_tag = penn_to_wn_tag(tag)
			if wn_tag not in (wn.ADJ, wn.ADV, wn.NOUN, wn.VERB):
				continue
			lemma = lemmatizer.lemmatize(word, pos=wn_tag)
			if not lemma:
				continue
			text = text.replace(word, lemma)
	return text

# Stemming text (Tested only, not used in the paper)
def stem_text(text):
	ps = nltk.porter.PorterStemmer()
	text = ' '.join([ps.stem(word) for word in text.split()])
	return text

# Subjectivity analysis
def subjective(text):
	raw_sentences = sent_tokenize(text)
	sid = SentimentIntensityAnalyzer()

	for raw_sentence in raw_sentences:		
		ss = sid.polarity_scores(raw_sentence)
		if ss['pos'] == 0 and ss['neg'] == 0:
			text = text.replace(raw_sentence," ")
	return text

# Help function to replace all chars in text
def replace_all(text, string_to_replace, string_to_replace_with):
	while string_to_replace in text:	
		text = text.replace(string_to_replace, string_to_replace_with)
	return text

# Functions to process contractions
# Initilaize the models
def init_contractions(api_key="glove-twitter-100"):
	cont = Contractions(api_key=api_key)
	cont.load_models()
	return cont

# Process contractions
def process_contractions(text, cont):
	if "'" in text:
		processed_text = list(cont.expand_texts([text], precise=True))
		text = "".join(processed_text)
	return text

# Process contraction in a dataste and export it to the same file
def process_dataset_cotractions(dataset_name = "processed_training"):
	data = load_dataset_from_csv(dataset_name)

	cont = init_contractions()

	keys = ["review_text", "comparative_sentences", "aspects_based_sentences"]

	data_length = len(data.index)

	for i in range(data_length):
		for k in keys:
			text = str(data[k][i])
			if text and text != "nan":
				processed_text = process_contractions(text, cont)
				data.loc[i,k] = processed_text
		loading(i+1, data_length, msg = "Processing contarctions in a total of "+str(data_length))
		
	data[["review_id", "review_key", "review_type", "review_text", "review_rating", "review_helpful", "sentiment_polarity_label", "is_comparative_label", "is_aspects_based_label", "comparative_sentences", "aspects_based_sentences", "review_status"]].to_csv("./datasets/"+dataset_name+".csv", sep="\t", quoting=3, index=False)

# Function  to export a dataset of pos tagged reviews (Used for testing and evaluation, not documented)
def export_pos_tags(dataset_name = "preprocessed_labeled", combined = False):
	data = load_dataset_from_csv(dataset_name)
	data_length = len(data.index)
	for i in range(data_length):
		review_text = str(data['review_text'][i])
		comparative_sentences = str(data['comparative_sentences'][i])
		aspects_based_sentences = str(data['aspects_based_sentences'][i])
		# Toknize and POS review_text
		review_text_tagged = to_pos_tags(review_text)
		comparative_sentences_tagged = ""
		aspects_based_sentences_tagged = ""

		# Toknize and POS comparative_sentences
		if comparative_sentences and comparative_sentences != "nan":
			comparative_sentences_tagged = to_pos_tags(comparative_sentences)

		# Toknize and POS aspects_based_sentences
		if aspects_based_sentences and aspects_based_sentences != "nan":
			aspects_based_sentences_tagged = to_pos_tags(aspects_based_sentences)
			
		if combined:
			review_text += ". "+ review_text_tagged.strip()
			comparative_sentences += ". "+ comparative_sentences_tagged.strip()
			aspects_based_sentences += ". "+ aspects_based_sentences_tagged.strip()
		else:
			review_text = review_text_tagged.strip()
			comparative_sentences = comparative_sentences_tagged.strip()
			aspects_based_sentences = aspects_based_sentences_tagged.strip()

		data.loc[i,"review_text"] = review_text
		data.loc[i,"comparative_sentences"] = comparative_sentences
		data.loc[i,"aspects_based_sentences"] = aspects_based_sentences

		loading(i+1, data_length, "Adding pos tags to a total of "+str(data_length)+" reviews")

	data[["review_id", "review_key", "review_type", "review_text",
		"review_rating", "review_helpful", "sentiment_polarity_label",
		"is_comparative_label", "is_aspects_based_label",
		"comparative_sentences", "aspects_based_sentences",
		"review_status"]].to_csv("./datasets/pos_tagged_"+dataset_name+".csv",
		sep="\t", quoting=3, index=False)

# Funtion to convert a PennTreebank tag to Wordnet tag
def penn_to_wn_tag(tag):
	if tag.startswith('J'):
		return wn.ADJ
	elif tag.startswith('N'):
		return wn.NOUN
	elif tag.startswith('R'):
		return wn.ADV
	elif tag.startswith('V'):
		return wn.VERB
	return None

# Add pos tags only
def to_pos_tags(text):
	clean_text = text.replace("/"," ").replace("-"," ").replace("_", " ").replace("?"," ").replace("!"," ")
	tokenized_text = nltk.word_tokenize(clean_text)
	tagged_text = nltk.pos_tag(tokenized_text)
	pos_tagged_text = ""
	for word, tag in tagged_text:
		if word == "." or word == ",":
			pos_tagged_text += str(word+" ")
		else:
			pos_tagged_text += str(tag+" ")
	return pos_tagged_text

# Loads a dataset from csv to pandas dataframe
def load_dataset_from_csv(dataset_name = "training"):
	data = pd.read_csv("./datasets/"+dataset_name+".csv", header=0, delimiter="\t", quoting=3, dtype={"review_id": str, "review_key": str, "review_type": str, "review_text": str, "review_rating": str, "review_helpful": str, "sentiment_polarity_label": str, "is_comparative_label": str, "is_aspects_based_label": str, "comparative_sentences": str, "aspects_based_sentences": str, "review_status": str})
	return data

# Loads a dataset from csv to pandas dataframe
def export_column_to_csv(dataset_name = "training", column_name="review_text"):
	data = load_dataset_from_csv(dataset_name)
	data[[column_name]].to_csv("./datasets/"+column_name+"_"+dataset_name+".csv", header=None, quoting=3, na_rep=' ', sep='\t', index=False)
	print("Exported successfully.")

# Show loading percentage
def loading(loaded, total, msg = "Loading"):
	percent = int((loaded/total)*100)
	sys.stdout.write(msg+" ("+str(loaded)+"):"+"%3d%%\r" % percent)
	sys.stdout.flush()

	if percent == 100:
		print (msg+" is 100% completed.")

# Database parameters (Change here for your environment)
# Note: Charset, Collation and Engine are recommended as it is, however, 
# pay special attention when chaging them if needed special data handling
database_host = "localhost"
database_port = 8889
database_user = "root"
database_password = "root"
database_name = "mlsaa"
database_charset = "utf8"
database_collation = "utf8_general_ci"
database_engine = "InnoDB"

# Returns a connection to the database
def db_connection(local_infile = 0):
	return pymysql.connect(host=database_host, 
							port=database_port, 
							user=database_user, 
							password=database_password, 
							db=database_name,
							local_infile =local_infile)

# Desc: Create the project mysql database 
def create_database():
	connection = pymysql.connect(host=database_host, 
							port=database_port, 
							user=database_user, 
							password=database_password)
	try:
		with connection.cursor() as cursor:
			# Note: database structure names can't be parametrized so it is concatinated to the sql string and other variables are given as paramters
			sql = "CREATE DATABASE IF NOT EXISTS `"+database_name+"` DEFAULT CHARACTER SET %s COLLATE %s"
			cursor.execute(sql, (database_charset, database_collation))
	finally:
		connection.close()

# Creates the datasets table
def create_dataset_table(table_name):
	connection = db_connection()
	try:
		with connection.cursor() as cursor:
			# Check if the table exisits to avoid any issues
			cursor.execute("SELECT `table_name` FROM INFORMATION_SCHEMA.TABLES WHERE `table_schema` = '"+database_name+"' AND `table_name` = '"+table_name+"'")
			if not cursor.fetchone():
				# Create the table statement
				sql = "CREATE TABLE `"+table_name+"` (`review_id` int(11) NOT NULL, `review_key` varchar(32) NOT NULL, `review_type` varchar(32) NOT NULL, `review_text` varchar(800) NOT NULL, `review_rating` varchar(32) NOT NULL, `review_helpful` varchar(32) NOT NULL, `sentiment_polarity_label` varchar(32) NOT NULL, `is_comparative_label` varchar(32) NOT NULL, `is_aspects_based_label` varchar(32) NOT NULL, `comparative_sentences` varchar(800) NOT NULL, `aspects_based_sentences` varchar(800) NOT NULL, `review_status` varchar(32) NOT NULL) ENGINE=%s DEFAULT CHARSET=%s;"
				result = cursor.execute(sql, (database_engine, database_charset))
				# Create a primary key
				cursor.execute("ALTER TABLE `"+table_name+"` ADD PRIMARY KEY (`review_id`);")
				# Create an auto increment property on the primary key
				cursor.execute("ALTER TABLE `"+table_name+"` MODIFY `review_id` int(11) NOT NULL AUTO_INCREMENT;")
	finally:
		connection.close()

# Creates the source table
def create_source_table(table_name):
	connection = db_connection()
	try:
		with connection.cursor() as cursor:
			# Check if the table exisits to avoid any issues
			cursor.execute("SELECT `table_name` FROM INFORMATION_SCHEMA.TABLES WHERE `table_schema` = '"+database_name+"' AND `table_name` = '"+table_name+"'")
			if not cursor.fetchone():
				sql = "CREATE TABLE `"+table_name+"` (`review_id` int(11) NOT NULL, `review_type` varchar(256) NULL, `reviewerID` varchar(32) NULL, `asin` varchar(256) NULL, `reviewerName` varchar(256) NULL, `helpful` varchar(32) NULL, `reviewText` varchar(3000) NULL, `overall` varchar(32) NULL, `summary` varchar(256) NULL, `unix_review_time` varchar(32) NULL, `review_time` varchar(32) NULL, `review_status` varchar(32) NULL) ENGINE=%s DEFAULT CHARSET=%s;"
				result = cursor.execute(sql, (database_engine, database_charset))
				# Create a primary key
				cursor.execute("ALTER TABLE `"+table_name+"` ADD PRIMARY KEY (`review_id`);")
				# Create an auto increment property on the primary key
				cursor.execute("ALTER TABLE `"+table_name+"` MODIFY `review_id` int(11) NOT NULL AUTO_INCREMENT;")
	finally:
		connection.close()

# Exports a dataset from databse to a csv file stored in datasets folder
def export_dataset(dataset_name = "training", quoting = False):
	# Get a database connection
	connection = db_connection()
	quote = '"'
	if not quoting:
		quote = ''
	# Export the dataset table into a csv file in the sources folder
	try:
		with connection.cursor() as cursor:
			cursor.execute("SELECT * FROM `"+dataset_name+"`")
			with open("./datasets/"+dataset_name+".csv", 'w') as f:
				header = quote+"review_id"+quote+"\t"+quote+"review_key"+quote+"\t"+quote+"review_type"+quote+"\t"+quote+"review_text"+quote+"\t"+quote+"review_rating"+quote+"\t"+quote+"review_helpful"+quote+"\t"+quote+"sentiment_polarity_label"+quote+"\t"+quote+"is_comparative_label"+quote+"\t"+quote+"is_aspects_based_label"+quote+"\t"+quote+"comparative_sentences"+quote+"\t"+quote+"aspects_based_sentences"+quote+"\t"+quote+"review_status"+quote+"\n"
				f.write(header)
				for result in cursor:
					review_text = remove_special_characters(str(result[3]))
					comparative_sentences = remove_special_characters(str(result[9]))
					aspects_based_sentences = remove_special_characters(str(result[10]))
					row = quote+str(result[0])+quote+"\t"+quote+str(result[1])+quote+"\t"+quote+str(result[2])+quote+"\t"+quote+review_text+quote+"\t"+quote+str(result[4])+quote+"\t"+quote+str(result[5])+quote+"\t"+quote+str(result[6])+quote+"\t"+quote+str(result[7])+quote+"\t"+quote+str(result[8])+quote+"\t"+quote+comparative_sentences+quote+"\t"+quote+aspects_based_sentences+quote+"\t"+quote+str(result[11])+quote+"\n"
					f.write(row)
	finally:
		connection.close()

# Import a csv / text file to database table
def import_dataset(dataset_name, table_name):
	if os.path.isfile("./datasets/"+dataset_name+".csv"):
		# Get database connection
		connection = db_connection(local_infile = 1)

		# Import the csv file
		try:
			with connection.cursor() as loader:
				loader.execute("TRUNCATE TABLE `"+table_name+"`")
				loader.execute("LOAD DATA LOCAL INFILE './datasets/"+dataset_name+".csv' INTO TABLE `"+table_name+"` FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' IGNORE 1 LINES (review_id, review_key, review_type, review_text, review_rating, review_helpful, sentiment_polarity_label, is_comparative_label, is_aspects_based_label, comparative_sentences, aspects_based_sentences, review_status) SET review_id = NULL")
			connection.commit()
		finally:
			connection.close()
	else:
		print("File doesn't exisit.")

# Exports a aource dataset from the source databse table to a csv file stored in datasets folder
def export_source(dataset_name = "source"):
	# Get a database connection
	connection = db_connection()

	# Export the dataset table into a csv file in the sources folder
	try:
		with connection.cursor() as cursor:
			cursor.execute("SELECT * FROM `"+dataset_name+"`")
			with open("./datasets/"+dataset_name+".csv", 'w') as f:
				header = "review_id\treview_type\treviewerID\tasin\treviewerName\thelpful\treviewText\toverall\tsummary\tunix_review_time\treview_time\treview_status\n"
				f.write(header)
				total_rows = cursor.rowcount
				added_rows = 0
				for result in cursor:
					reviewer_name = remove_empty_lines(str(result[4]))
					review_text = remove_empty_lines(str(result[6]))
					review_summary = remove_empty_lines(str(result[8]))
					row = str(result[0])+"\t"+str(result[1])+"\t"+str(result[2])+"\t"+str(result[3])+"\t"+reviewer_name+"\t"+str(result[5])+"\t"+review_text+"\t"+str(result[7])+"\t"+review_summary+"\t"+str(result[9])+"\t"+str(result[10])+"\t"+str(result[11])+"\n"
					f.write(row)
					added_rows += 1
					loading(added_rows, total_rows, "Exporting a total of "+str(total_rows)+" from source")
	finally:
		connection.close()

# Import the source csv / text file to the database table
# The code to convert the source files from JSON to CSV is given in
# the source dataset link, so it was not included in this module.
def read_source(dataset_name = "source", table_name = "source"):
	"""The function reads the source dataset from source.csv file and loads
			it in the database.

    Args:
        dataset_name (str): The name of the source dataset (the name of CSV file).
        table_name (str): The table name to load the source dataset to it.

    Returns:
        None
    """
	if os.path.isfile("./datasets/"+dataset_name+".csv"):
		# Get database connection
		connection = db_connection(local_infile = 1)

		# Loads the source csv file
		try:
			with connection.cursor() as loader:
				loader.execute("TRUNCATE TABLE `"+table_name+"`")
				loader.execute("LOAD DATA LOCAL INFILE './datasets/"+dataset_name+".csv' INTO TABLE `"+table_name+"` FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n' IGNORE 1 LINES (review_id, review_type, reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unix_review_time, review_time, review_status)")
			connection.commit()
		finally:
			connection.close()
	else:
		print("File doesn't exisit.")

# Load 200,000 records into the development dataset 
# Creates a csv file for the dataset and loads it to the database
# total_rows = total required rows to be loaded
def prepare_development(total_rows = 200000, min_review_length = 200, max_review_length = 700): 
	"""The function prepares the development dataset from the loaded source dataset
	 and stores it in a database for manual labeling.

    Args:
        total_rows (int): The total number of reviews selected fot development.
        min_review_length (int): The minimum review length.
        max_review_length (int): The maximum review length.

    Returns:
        None
    """
	# init teh loaded rows number and default to 0
	loaded_rows = 0
	selected_ids = []

	# *** Note: If a labeled dataset is found in the datasets folder in parent directoy
	# it will be loaded first to the development table and then the table is filled up to
	# the number of total rows (This was needed when preparing the devlopment dataset 
	# again during labeling).
	source_path = r"./datasets/source.csv"
	development_path = r"./datasets/development.csv"
	training_path = r"./datasets/labeled.csv"
	selected_path = r"./datasets/selected.csv"

	# Check if a processed labeled dataset is there
	if os.path.isfile("./datasets/processed_labeled.csv"):
		training_path = "./datasets/processed_labeled.csv"

	# Get the already labeled reviews
	already_labeled_data = pd.read_csv(training_path, header=0, delimiter="\t", quoting=3,
	 									dtype={"review_id": str, "review_key": str,
	 									"review_type": str, "review_text": str, 
	 									"review_rating": str, "review_helpful": str,
	 									"sentiment_polarity_label": str,
	 									"is_comparative_label": str,
	 									"is_aspects_based_label": str,
	 									"comparative_sentences": str, 
	 									"aspects_based_sentences": str, 
	 									"review_status": str})

	# Add the ids of the already labeled reviews to the selcted ids array
	selected_rows = len(labeled_data.index)
	for i in range(selected_rows):
		selected_ids.append(str(already_labeled_data["review_id"][i]))

	# Get the previously selected reviews if there or in case if prepartion was interupted.
	# This is done due to the long time needed to prepare the development dataset, so interuptions
	# may happen before the devleopment dataset is exported into CSV or loaded to database.
	# Therefore, it was decided to save an array of selected ids in a seperate CSV file.
	# It is alos saved every 100 loaded reviews.
	previously_selected_data = pd.read_csv(selected_path, header=0, delimiter="\t", quoting=3,
										dtype={"review_id": str, "review_key": str,
										"review_type": str, "review_text": str,
										"review_rating": str, "review_helpful": str,
										"sentiment_polarity_label": str, 
										"is_comparative_label": str, 
										"is_aspects_based_label": str, 
										"comparative_sentences": str, 
										"aspects_based_sentences": str, 
										"review_status": str})

	# Add the ids of the previously selected reviews to the selcted ids array
	selected_rows += len(previously_selected_data.index)
	for i in range(selected_rows):
		selected_ids.append(str(previously_selected_data["review_id"][i]))

	# Adds the data to teh development dataset CSV file
	already_labeled_data.to_csv(development_path, header=0, quoting=3, index=None, sep='\t', mode='w')
	previously_selected_data.to_csv(development_path, header=0, quoting=3, index=None, sep='\t', mode='a')

	# Reads the source data
	source_data = pd.read_csv(source_path, header=0, delimiter="\t", quoting=3, dtype={"review_id": str,
							"review_type": str, "reviewerID": str, "asin": str, "reviewerName": str,
							"helpful": str, "reviewText": str, "overall": str, "summary": str, 
							"unix_review_time": str, "review_time": str, "review_status": str})

	# Prepare a dataframe for new selected data
	selected_data = pd.DataFrame(dtype={"review_id": str, "review_key": str, "review_type": str, 
							"review_text": str, "review_rating": str, "review_helpful": str, 
							"sentiment_polarity_label": str, "is_comparative_label": str, 
							"is_aspects_based_label": str, "comparative_sentences": str, 
							"aspects_based_sentences": str, "review_status": str})

	# Init contractions model
	cont = init_contractions()

	# Save the review rating during iterations to make sure we are getting a normal distribution over positive and negative reviews 
	positive = 1

	# A temporaryy int to count to 100 and append the seleted rows
	append = 0

	# Fill development table while less than total number of rows
	while selected_rows < total_rows:
		# Select 100 reviews from source table where review id (To check for the coming conditions while making use of one select) is larger than a random number and overall rating not equal to 3.0 (As we skipp neutral)
		# Note: this is to achieve good distribution the rating need to vary each time insreted in the table, depending on the balance argument
		index = random.randint(0,3250000)
		limit = index+100

		# Loop the results searching for quality and shor reviews
		for i in range(index, limit):
			data_list = source_data.values[i].tolist()

			# Check if the review is already Selected
			review_id = data_list[0]
			if review_id in selected_ids:
				continue

			# start = time.time()
			review_rating = data_list[7]

			# Skip neutral reviews
			if review_rating == "3.0":
				continue

			# Balancing reviews between negative and positive
			if balance:
				if positive and (review_rating == "1.0" or review_rating == "2.0"):
					continue
				if not positive and (review_rating == "4.0" or review_rating == "5.0"):
					continue

			review_text = data_list[6] # Text is cleaned during loading
			review_length = len(review_text)

			# Filter by review length and helpful to get higher quality reviews
			if min_review_length < review_length < max_review_length:
				review_type = data_list[1]
				review_key = review_id+"_"+data_list[2]+"_"+data_list[3]+"_"+data_list[9]
				review_helpful = data_list[5]
				summary = data_list[8]
				
				# Concatinate the summary with the review
				review_text = summary.strip(".") + ". " + review_text
				review_text = process_text(review_text)
				review_text = process_contractions(review_text, cont)

				#Calculate the sentiment polarity based on the overall rating
				sentiment_polarity_label = "Positive" if positive else "Negative"

				# 
				selected_data = selected_data.append({"review_id": selected_rows+1, "review_key": review_key, 
												"review_type": review_type, "review_text": review_text, 
												"review_rating": review_rating, "review_helpful": review_helpful, 
												"sentiment_polarity_label": sentiment_polarity_label, 
												"is_comparative_label": "No", "is_aspects_based_label": "No", 
												"comparative_sentences": "", "aspects_based_sentences": "", 
												"review_status": "New"}, ignore_index=True)

				# Append the selected 100 reviews to the development dataset
				if append == 100:
					selected_data.to_csv(development_path, header=0, quoting=3, index=None, sep='\t', mode='a')
					selected_data = pd.DataFrame(dtype={"review_id": str, "review_key": str, "review_type": str, "review_text": str, "review_rating": str, "review_helpful": str, "sentiment_polarity_label": str, "is_comparative_label": str, "is_aspects_based_label": str, "comparative_sentences": str, "aspects_based_sentences": str, "review_status": str})
					append = 0
				else:
					append+=1
				
				# Update loop variables
				selected_rows +=1
				loading(loaded_rows, total_rows, msg = "Loading a total of "+str(total_rows)+" reviews into development")
				positive = not positive
				selected_ids.append(review_id)
				break

	# Append the selected reviews to the development dataset one last time
	# At this stage the complete development dataset is exported to CSV successfully
	selected_data.to_csv(development_path, header=0, quoting=3, index=None, sep='\t', mode='a')
	import_dataset("development", "development")

	# Save the devleopment dataset in the database
	connection = db_connection()
	try:
		with connection.cursor() as cursor:
			updated_rows = 0
			for selected_id in selected_ids:
				sql = "UPDATE `source` SET `review_status` = 'Selected' WHERE `review_id` = %s"
				cursor.execute(sql, (selected_id, ))
				updated_rows+=1
				loading(updated_rows, total_rows, msg = "Updating a total of "+str(total_rows)+" source reviews to selected status ("+str(updated_rows)+")")
		connection.commit()
	finally: 
		connection.close()

	# Exported an upadted source CSV file with status update for the selected reviews for development
	export_source()

# Help fucntion to plot the confusion matrix of the classifications
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=3)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Help function that returns an iteratble from gzipped file
def parse_gz(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield eval(l)

# Help function that returns a dataframe from dectionary
def get_df(path):
	i = 0
	df = {}
	for d in parse_gz(path):
		df[i] = d
		i += 1
	return pd.DataFrame.from_dict(df, orient='index')

# Help function required for transforming datasets for classification
class DenseTransformer(TransformerMixin):

	def transform(self, X, y=None, **fit_params):
		return X.todense()	

	def fit_transform(self, X, y=None, **fit_params):
		self.fit(X, y, **fit_params)
		return self.transform(X)

	def fit(self, X, y=None, **fit_params):
		return self

"""
Scikit-learn License:
	New BSD License

	Copyright (c) 2007–2018 The scikit-learn developers.
	All rights reserved.


	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	  a. Redistributions of source code must retain the above copyright notice,
	     this list of conditions and the following disclaimer.
	  b. Redistributions in binary form must reproduce the above copyright
	     notice, this list of conditions and the following disclaimer in the
	     documentation and/or other materials provided with the distribution.
	  c. Neither the name of the Scikit-learn Developers  nor the names of
	     its contributors may be used to endorse or promote products
	     derived from this software without specific prior written
	     permission. 


	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
	OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
	DAMAGE.

scikit-multilearn License:
	New BSD License

	Copyright (c) 2007–2018 The scikit-learn developers.
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, 
	are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, 
	this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright notice,
	this list of conditions and the following disclaimer in the documentation and/or
	other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
	LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
	CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
	SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
	INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
	POSSIBILITY OF SUCH DAMAGE.

NLTK License:
	Copyright (C) 2001-2019 NLTK Project

	Licensed under the Apache License, Version 2.0 (the 'License');
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an 'AS IS' BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

References:
	@article{2017arXiv170201460S,
		author = {{Szyma{\'n}ski}, P. and {Kajdanowicz}, T.},
		title = "{A scikit-based Python environment for performing multi-label classification}",
		journal = {ArXiv e-prints},
		archivePrefix = "arXiv",
		eprint = {1702.01460},
		primaryClass = "cs.LG",
		keywords = {Computer Science - Learning, Computer Science - Mathematical Software},
		year = 2017,
		month = feb,
    }
    @inproceedings{read2009classifier,
		title={Classifier chains for multi-label classification},
		author={Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe},
		booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
		pages={254--269},
		year={2009},
		organization={Springer}
	}
	@article{5567103,
		author={G. Tsoumakas and I. Katakis and I. Vlahavas},
		journal={IEEE Transactions on Knowledge and Data Engineering},
		title={Random k-Labelsets for Multilabel Classification},
		year={2011},
		volume={23},
		number={7},
		pages={1079-1089},
		doi={10.1109/TKDE.2010.164},
		ISSN={1041-4347},
		month={July},
	}
	@article{zhang2007ml,
		title={ML-KNN: A lazy learning approach to multi-label learning},
		author={Zhang, Min-Ling and Zhou, Zhi-Hua},
		journal={Pattern recognition},
		volume={40},
		number={7},
		pages={2038--2048},
		year={2007},
		publisher={Elsevier}
	}
	@article{scikit-learn,
		title={Scikit-learn: Machine Learning in {P}ython},
		author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
		     and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
		     and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
		     Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
		journal={Journal of Machine Learning Research},
		volume={12},
		pages={2825--2830},
		year={2011}
	}
	@book{NLTK,
		title={Natural Language Processing with Python,
		author={Steven Bird, Edward Loper, Ewan Klein},
		publisher={O’Reilly Media Inc.}
		year={2009}
	}
"""