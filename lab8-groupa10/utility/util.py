import numpy as np
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.utils import Bunch


def configure_plots():
    '''Configures plots by making some quality of life adjustments'''

    for _ in range(2):
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['lines.linewidth'] = 2

def distance_measure(a, b):
    '''A measures a distance between point(s) a and b.'''

    return np.linalg.norm(a - b, axis=int(len(a.shape) > 1))

def plt_data(X_2D_train, y_train, encoding, X_test, y_test_pred=None):
    for species, code in encoding.items():
        train_idx = y_train == code
        p = plt.scatter(X_2D_train[train_idx,0], X_2D_train[train_idx,1], label=f'{species} ({code})')

        if y_test_pred is not None:
            test_idx = y_test_pred == code
            color = p.get_facecolor()
            plt.scatter(X_test[test_idx,0], X_test[test_idx,1], color=color, marker='*', s=120, label=f'predicted {species} ({code})')

    if y_test_pred is None:
        test_idx = np.ones(X_test.shape[0], dtype=bool)
        plt.scatter(X_test[test_idx,0], X_test[test_idx,1], color='gray', marker='*', s=120, label=f'test point')

    plt.xlabel("Sepal Width")
    plt.ylabel("Sepal Length")
    plt.legend()
    plt.title('X_2D_train, class labels, and test point(s)')
    plt.show()


def plot_knn(X_train, X_test, y_train, y_test, k=3, title=None):
    from sklearn import neighbors
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    #plt.rcParams['figure.figsize'] = [12, 8]

    #specify classifier
    clf = neighbors.KNeighborsClassifier(k)

    #fit our data
    clf.fit(X_train, y_train)

    # setosa, versicolor, virginica
    list_dark = ['#08519c','#a63603','#006d2c']
    cmap_dark = ListedColormap(list_dark)
    list_bold = ['#3182bd','#e6550d','#31a354']
    cmap_bold = ListedColormap(list_bold)
    list_light = ['#bdd7e7','#fdbe85','#bae4b3']
    cmap_light = ListedColormap(list_light)

    # calculate min, max and limits
    X = np.concatenate((X_train, X_test))
    h = 0.02
    #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # predict class using data and kNN classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap = cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Sepal Width")
    plt.ylabel("Sepal Length")
    legend_elements = [
        Line2D([0], [0], marker='o', color=list_dark[0], label='setosa (prediction region)',
               markerfacecolor=list_light[0], markersize=10),
        Line2D([0], [0], marker='o', color=list_dark[1], label='versicolor (prediction region)',
               markerfacecolor=list_light[1], markersize=10),
        Line2D([0], [0], marker='o', color=list_dark[2], label='virginica (prediction region)',
               markerfacecolor=list_light[2], markersize=10),
        Line2D([0], [0], linewidth=0, marker='X', color='gray', label='test point (ground truth color)',
                markerfacecolor='gray', markersize=15),
    ]

    # Plot also the test points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='X', s=160, cmap=cmap_dark)

    acc = clf.score(X_test,y_test)
    if title==None:
        plt.title("kNN with k = %i has %.2f%% accuracy" % (k, acc*100))
    else:
        plt.title(title+" kNN with k = %i has %.2f%% accuracy" % (k, acc*100))
        
    plt.legend(handles = legend_elements)
    plt.show()
    
def load_boston():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    return Bunch(
        data=data,
        target=target,
        feature_names=np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
        'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7'),
        DESCR=""".. _boston_dataset:\n\nBoston house prices dataset\n---------------------------\n\n**Data Set Characteristics:**  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n.. topic:: References\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n"""
    )
