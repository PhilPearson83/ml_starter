# Compare Algorithms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

h = .02  # step size in the mesh

# Disable annoying warnings
import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=UserWarning)

# Performance
import time

# load dataset into pandas dataframe
inputfile = 'W:/2_Reference_Materials/Python/BusSafetyCompliance/Comp.csv'
y_field = 'Job_Outcome'
df = pd.read_csv(inputfile)
df = df.iloc[:, 0:8]

# create x and y values
x = np.array(df.drop(y_field, axis=1))
y = np.array(df[y_field])

# set seed for consistancy
seed = 123

# create list of models
models = []
#models.append(('LinReg', LinearRegression()))
#models.append(('LinReg', Ridge()))
#models.append(('LinReg', Lasso()))
#models.append(('BR', BayesianRidge()))
models.append(('SGD', SGDClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RNC', RadiusNeighborsClassifier(radius = 200)))
models.append(('DecTree', DecisionTreeClassifier(max_depth=5)))
models.append(('NB', GaussianNB()))
#models.append(('CB', ComplementNB())) # Can't have negative X values
models.append(('SVM', SVC()))
models.append(('RanFor', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
models.append(('ExTree', ExtraTreesClassifier()))
models.append(('Ada', AdaBoostClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
#models.append(('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))))
#models.append(('Neural Net', MLPClassifier(alpha=1, max_iter=1000)))

# evaluate each model in the list
results = []
names = []
scoring = 'accuracy'
print("---------------------------------------")
for name, model in models:
    start_time = time.time()
    kfold = model_selection.KFold(n_splits=15, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    elapsed_time = time.time() - start_time
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f) Time elapsed: %f" % (name, cv_results.mean(), cv_results.std(), elapsed_time)
    print(msg)
print("---------------------------------------")

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylim((None,1))
plt.show()   

# split dataframe into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


figure = plt.figure(figsize=(18, 9))
i = 1
#cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(len(models), len(models) + 1, i)
ax.set_title("Input data")
# Plot the training points
ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
# Plot the testing points
#ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
#plt.tight_layout()
plt.show()