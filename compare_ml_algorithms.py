import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, Ridge, Lasso, BayesianRidge, SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

# Disable annoying warnings
import warnings
warnings.filterwarnings('ignore')
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=UserWarning)

h = .02  # step size in the mesh
seed = 123 # set random number for consistancy

# load dataset into dataframe
inputfile = 'W:/2_Reference_Materials/Python/BusSafetyCompliance/Comp.csv'
y_value = 'Job_Outcome'
df = pd.read_csv(inputfile)
df = df.iloc[:, 0:3]


# create x y values and standardisation x array
x = df.drop(y_value, axis=1)
feature_names = list(x.columns.values)
std_scale = StandardScaler()
X_std = np.array(std_scale.fit_transform(x))
Y = np.array(df[y_value])

# create x and y values
x = np.array(df.drop(y_value, axis=1))
y = np.array(df[y_value])


# create list of models
models = []
#models.append(('LinReg', LinearRegression()))
#models.append(('LinReg', Ridge()))
#models.append(('LinReg', Lasso()))
#models.append(('BR', BayesianRidge()))
models.append(('SGD', SGDClassifier(max_iter=1000, random_state=seed, class_weight='balanced')))
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('LRCV', LogisticRegressionCV(class_weight='balanced')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('RNC', RadiusNeighborsClassifier(radius = 200)))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced'))) # max_depth=5
models.append(('RanFor', RandomForestClassifier(class_weight='balanced'))) # max_depth=5, n_estimators=10, max_features=1
models.append(('ExTree', ExtraTreesClassifier(class_weight='balanced')))
models.append(('BC', BaggingClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('CB', ComplementNB())) # Can't have negative X values
models.append(('SVM', SVC(class_weight='balanced')))
models.append(('Ada', AdaBoostClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
#models.append(('Gaussian Process', GaussianProcessClassifier(1.0 * RBF(1.0))))
models.append(('MLP', MLPClassifier(alpha=1, max_iter=1000)))

# evaluate each model in the list
results = []
names = []
scoring = 'accuracy'
print("---------------------------------------")
for name, model in models:
    start_time = time.time()
    kfold = model_selection.KFold(n_splits=15, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_std, Y, cv=kfold, scoring=scoring)
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

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)

# standardisation of x arrays
std_scale = StandardScaler()
x_train_scaled = std_scale.fit_transform(x_train) 
x_test_scaled = std_scale.transform(x_test) 

#x_min, x_max = x_train_scaled[:, 0].min() - .5, x_train_scaled[:, 0].max() + .5
#y_min, y_max = x_train_scaled[:, 1].min() - .5, x_train_scaled[:, 1].max() + .5
x_min, x_max = x_train_scaled[:, 0].min() - .5, x_train_scaled[:, 0].max() + .5
y_min, y_max = x_train_scaled[:, 1].min() - .5, x_train_scaled[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


figure = plt.figure(figsize=(27, 9))
i = 1
cm = plt.cm.get_cmap('RdBu')
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(models) + 1, i)
ax.set_title("Input data")

# Plot the training points
ax.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

# Plot the testing points
#ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, cmap=cm_bright, alpha=0.3, edgecolors='k')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in models:
    ax = plt.subplot(1, len(models) + 1, i)
    clf.fit(x_train_scaled, y_train)
    score = clf.score(x_test_scaled, y_test)

    # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.3)

    # Plot the training points
    #ax.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, cmap=cm_bright, alpha=0.3, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    i += 1
plt.tight_layout()
plt.show()
