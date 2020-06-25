import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from operator import itemgetter

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
seed = 123  # set random number for consistancy
testsize = 0.20  # split value for test

# load dataset into dataframe
inputfile = 'W:\\2_Reference_Materials\\Python\\BusSafetyCompliance\\Comp.csv'
y_value = 'Job_Outcome'
df = pd.read_csv(inputfile)
df = df.iloc[:, 0:3]

# create x y values, standardise x and fit to y
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
#models.append(('LRCV', LogisticRegressionCV(class_weight='balanced'))) #The object works in the same way as GridSearchCV except that it defaults to Generalized Cross-Validation (GCV), an efficient form of leave-one-out cross-validation
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('SVM_lin', SVC(kernel='linear')))
models.append(('SVM_rbf', SVC(kernel='rbf')))
# models.append(('SVM_sig', SVC(kernel='sigmoid', class_weight='balanced'))) # Y value the wrong way around??
models.append(('SVM_poly', SVC(kernel='poly')))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('RNC', RadiusNeighborsClassifier(radius=10, weights='uniform')))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))  # max_depth=5 # max_depth=5, n_estimators=10, max_features=1
models.append(('RanFor', RandomForestClassifier(class_weight='balanced')))
models.append(('ExTree', ExtraTreesClassifier(class_weight='balanced')))
models.append(('BC', BaggingClassifier()))
models.append(('NB', GaussianNB()))
# models.append(('CB', ComplementNB())) # Can't have negative X values
models.append(('Ada', AdaBoostClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
# models.append(('GPC', GaussianProcessClassifier())) #kernel = 1.0 * RBF(1.0)
#models.append(('MLP', MLPClassifier(alpha=0.0001, max_iter=1000)))

# evaluate each model in the list
combined_results = []
scoring = ['accuracy']
print('---------------------------------------')
for name, model in models:
    start_time = time.time()
    kfold = model_selection.KFold(n_splits=2, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_std, Y, cv=kfold, scoring=scoring)
    elapsed_time = time.time() - start_time
    # results.append(cv_results)
    # names.append(name)
    combined_results.append((name, cv_results.mean(), cv_results))
    msg = "%s: %f (%f) Time elapsed: %f" % (name, cv_results.mean(), cv_results.std(), elapsed_time)
    print(msg)
print('---------------------------------------')

# sort our results by mean avg score and assign results to new variable
combined_results_sorted = sorted(combined_results, key=itemgetter(1,0))
modelnamme, meanscore, results = zip(*combined_results_sorted)
#labels, meanscore, results = [i[0] for i in combined_results_sorted], [i[1] for i in combined_results_sorted], [i[2] for i in combined_results_sorted]

# boxplot the algorithm comparison
fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax.set_xticklabels(modelnamme)
orange_circle = dict(markerfacecolor='orange', marker='o')
plt.boxplot(x=results, flierprops=orange_circle)  # notch=True bootstrap=1000
plt.ylim((None, 1))
plt.xlabel('Model')
plt.title('Model Comparison \n (kfold = ' + str(kfold.n_splits) + ')')
plt.show()

# split data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=seed)

# standardisation of x arrays
std_scale = StandardScaler()
x_train_scaled = std_scale.fit_transform(x_train)
x_test_scaled = std_scale.transform(x_test)

#x_min, x_max = x_train_scaled[:, 0].min() - .5, x_train_scaled[:, 0].max() + .5
#y_min, y_max = x_train_scaled[:, 1].min() - .5, x_train_scaled[:, 1].max() + .5
x_min, x_max = x_train_scaled[:, 0].min() - .5, x_train_scaled[:, 0].max() + .5
y_min, y_max = x_train_scaled[:, 1].min() - .5, x_train_scaled[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# set variable to iterate chart subplots
i = 1
# set some colour variables to use and labels
cm = plt.cm.get_cmap('RdBu_r') 
cm_bright = ListedColormap(['blue', 'red'])
labels = ['Sat', 'Unsat']
# create plot area
fig = plt.figure(figsize=(27, 9))
ax = plt.subplot(1, len(models) + 1, i)
# ax = plt.subplot(1, len(models) + 1, i))
# set title and min / max values for axis
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Input data")
# Plot the training points
scatter = ax.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap=cm_bright)#, edgecolors='k')
#plt.legend(*scatter.legend_elements(),loc="upper right")
plt.legend(handles=scatter.legend_elements()[0], labels=labels, loc="upper right")
# Plot the testing points
#ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, cmap=cm_bright, alpha=0.3, edgecolors='k')
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
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.25)

    # Plot the training points
    #ax.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    #ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, cmap=cm_bright, alpha=0.3, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
    i += 1
plt.tight_layout()
plt.show()
