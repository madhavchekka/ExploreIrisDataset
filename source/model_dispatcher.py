# model_dispatcher.py
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model

models = {
	"decision_tree_gini": tree.DecisionTreeClassifier(
		criterion="gini",
	),
	"decision_tree_entropy": tree.DecisionTreeClassifier(
		criterion="entropy",
	),
	"rf": ensemble.RandomForestClassifier(n_jobs=-1),
	"logistic" : linear_model.LogisticRegression(n_jobs=-1,
		solver='lbfgs'
	)
}