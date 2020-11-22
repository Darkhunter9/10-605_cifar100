
from sklearn.tree import export_graphviz
# Export as dot file
import pickle
import os

from os import listdir
from os.path import isfile, join

mypath = "./"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

filenames = onlyfiles



for filename in filenames:

	model = None

	if os.path.exists(filename) and filename.endswith(".rfmodel") :
		with open(filename, "rb") as file:
			model = pickle.load(file)

	if model is None:
		continue

	# Extract single tree
	estimator = model.estimators_[0]

	export_graphviz(estimator, out_file='tree.dot', 
					# feature_names = iris.feature_names,
					# class_names = iris.target_names,
					rounded = True, proportion = False, 
					precision = 2, filled = True)

	# Convert to png using system command (requires Graphviz)
	from subprocess import call
	call(['dot', '-Tpng', 'tree.dot', '-o', f"{filename}_RF_tree_vis.png", '-Gdpi=600'])

	# Display in jupyter notebook
	from IPython.display import Image
	Image(filename = 'tree.png')
