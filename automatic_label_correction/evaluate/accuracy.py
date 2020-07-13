from ..utils import KFold, Falsify
from sklearn.neighbors import KNeighborsClassifier as NNC

def Accuracy(p, number_of_runs, method, clf = None, X, y):

    if clf is None:
        clf = NNC()

    acc = np.zeros((number_of_runs, 2))
    for _ in range(number_of_runs):

        # Artificially falsify
        y_f = Falsify(y, p)

        # Correct labels
        y_corrected = method.fit_transform(X, y_f)

        # Set up 10-fold-Cross validation
        train_corr, test_corr = kFold(10, X, y, y_corrected)
        train_f, test_f = kFold(10, X, y, y_f)

        score = np.zeros((2, 10))

        # Calc scores
        for fold in range(10):

            train_X, train_y = train_corr[fold]
            test_X, test_y = test_corr[fold]
            clf.fit(train_X, train_y)
            score[0, fold] = clf.score(test_X, test_y)

            train_X, train_y = train_f[fold]
            test_X, test_y = test_f[fold]
            clf.fit(train_X, train_y)
            score[1, fold] = clf.score(test_X, test_y)

        # Average
        acc[_] = np.mean(score, axis = 1)

    return np.mean(acc, axis = 0), np.std(acc, axis = 0)
