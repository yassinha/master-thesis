
# Master Thesis Work

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
	Jönköping, Sweden. December 2018.


## About mlsaa.py module

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
	
	get_reviews: Gets the reviews text array after preprocessing.
	
	get_labels: Gets the labels array that is transformed into a multi-label format.
	
	classify: Run the classification model which includes the multi-label 
	classification technique, machine learning classifier, dataset sampling,
	features selection, and evaluation functions.

Example:
	
	$ python
	
	>>> import mlsaa
        
	>>> mlsaa.classify(d="dataset_name", ct=1, pt=1, cl=1)
	
## About run.py module

This module is for running the experiments in the paper. 

Example:
	To run experiment EXP10 as in the thesis
	
	$ python run.py EXP10

