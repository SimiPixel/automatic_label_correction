from ..utils import kfold, falsify
from sklearn.neighbors import KNeighborsClassifier as NNC
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from pathos.multiprocessing import cpu_count

def accuracy(p, number_of_runs, method, X, y, clf = None, n_jobs = None):
    '''
    Returns: (array{mean_acc_corrected, mean_acc_false}, array{stddev_acc_corrected, stddev_acc_false})
    '''

    # Setup parallel job
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs == None:
        n_jobs = 1

    pool = Pool(n_jobs, maxtasksperchild = 1000)


    if clf is None:
        clf = NNC()

    def run(_):

        # Artificially falsify
        y_f = falsify(y, p, random_state = _)

        # Correct labels
        y_corrected = method.fit_transform(X, y_f)

        # Set up 10-fold-Cross validation
        train_corr, test_corr = kfold(10, X, y, y_corrected)
        train_f, test_f = kfold(10, X, y, y_f)

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
        return np.mean(score, axis = 1).tolist()

    acc = np.array(pool.map(run, range(number_of_runs)))

    # Close the pool again
    pool.close()
    pool.join()
    pool.clear()

    return np.mean(acc, axis = 0), np.std(acc, axis = 0)
