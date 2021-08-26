import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def model(X_train, Y_train, X_test, Y_test):
    X = X_train.values.reshape(-1, 1)
    Y = Y_train.values.reshape(-1, 1)

    model = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
    model.fit(X, Y)
    print(model.score(X, Y))
    train_score = model.score(X, Y)

    plt.scatter(X_train, Y_train)
    plt.scatter(X_train, model.predict(X), color='red')
    plt.show()

    X = X_test.values.reshape(-1, 1)
    Y = Y_test.values.reshape(-1, 1)
    print(model.score(X, Y))
    test_score = model.score(X, Y)

    plt.scatter(X_test, Y_test)
    plt.scatter(X_test, model.predict(X), color='red')
    plt.show()

    if test_score >= train_score:
        print('Pass')

    else:
        print('Fail')
