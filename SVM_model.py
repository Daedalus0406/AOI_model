import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

def model_OC (X_train, X_test):

    # fit the model (nu=0.1)
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
    clf.fit(X_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(y_pred_test)

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


    """
    Visualization of the result.
   
    xx, yy = np.meshgrid(np.linspace(0, 300, 500), np.linspace(-1, 1, 500))

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10,6))
    plt.title('Novelty Detection')
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s, edgecolors='k')

    plt.axis('tight')
    plt.xlim((0, 300))
    plt.ylim((-1, 1))
    plt.legend([a.collections[0], b1, b2, c],
               ['learned frontier', 'training observations',
                'new regular observations', 'new abnormal observations'],
               loc='upper left',
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        'train: %.3f; test: %.3f; out: %.3f'
        % (error_train, error_test, error_outliers))

    plt.show()
    """
    return y_pred_train, y_pred_test

def model_SVC (X_train, Y_train, X_test):

    # fit the model (nu=0.1)
    clf = svm.SVC(kernel = 'linear', random_state = 0)
    clf.fit(X_train, Y_train)

    y_pred_test = clf.predict(X_test)
    print(y_pred_test)


