import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def model (X_train, X_test):

    # fit the model
    clf = IsolationForest(max_samples=1000)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_pas_train  = y_pred_train[y_pred_train == 1].size
    train_total = n_error_train + n_pas_train
    error_train = n_error_train / train_total
    print(error_train)

    n_error_test = y_pred_test[y_pred_test == -1].size
    n_pas_test = y_pred_test[y_pred_test == 1].size
    test_total = n_error_test + n_pas_test
    error_test = n_error_test/test_total
    print(error_test)


    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(0, 300, 500), np.linspace(-1, 1, 500))

    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')

    plt.axis('tight')
    plt.xlim((0, 300))
    plt.ylim((-1, 1))
    plt.legend([b1, b2],
               ["training observations",
                "new regular observations"],
               loc="upper left")
    plt.show()