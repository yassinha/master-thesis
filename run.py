import mlsaa
import sys

# A configuration for prediction
experiments = ["EXP1", "EXP2",
	"EXP3", "EXP4",
	"EXP5", "EXP6",
	"EXP7", "EXP8",
	"EXP9", "EXP10",
	"EXP11", "EXP12",
	"EXP13", "EXP14",
	"EXP15","EXP16"]

# A variable for the experiment code to run
experiment = "EXP1"

# Reads the target experiment from system argument or input line is set
for i in range(1,len(sys.argv)):
	arg = sys.argv[i]
	if arg is not None:
		if arg in experiments:
			experiment = arg
		else:
			print("Run Argument  should be a valid experiment code 'EXP1-EXP16'.")
	else:
		print("Command ignored and default values used.")

if experiment == "EXP1":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=1, cl=1)
elif experiment == "EXP2":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=1, cl=2)
elif experiment == "EXP3":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=1, cl=3)
elif experiment == "EXP4":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=2, cl=1)
elif experiment == "EXP5":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=2, cl=2)
elif experiment == "EXP6":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=2, cl=3)
elif experiment == "EXP7":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=3, cl=1)
elif experiment == "EXP8":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=3, cl=2)
elif experiment == "EXP9":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=3, cl=3)
elif experiment == "EXP10":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=0, cl=4)
elif experiment == "EXP11":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=0, cl=5)
elif experiment == "EXP12":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=1, pt=0, cl=6, fo=3)
elif experiment == "EXP13":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=2, pt=0, cl=7)
elif experiment == "EXP14":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=3, pt=0, cl=1)
elif experiment == "EXP15":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=3, pt=0, cl=2)
elif experiment == "EXP16":
	mlsaa.classify(dataset_name="preprocessed_labeled", ct=3, pt=0, cl=3)

