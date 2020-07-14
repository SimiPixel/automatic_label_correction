from ..utils import Falsify
import numpy as np
from pathos.multiprocessing import ProcessPool as Pool
from pathos.multiprocessing import cpu_count


def CorrectionFactor(p, number_of_runs, method, X, y, n_jobs = None):

    # Setup parallel job
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs == None:
        n_jobs = 1

    pool = Pool(n_jobs, maxtasksperchild = 1000)

    def run(_):

        # Artificially falsify
        y_f = Falsify(y, p, random_state = _)

        # Correct labels
        y_corrected = method.fit_transform(X, y_f)

        N = X.shape[0]
        return ((y == y_corrected).sum() - (1-p)*N)/(p*N)

    factor = np.array(pool.map(run, range(number_of_runs)))

    # Close the pool again
    pool.close()
    pool.join()
    pool.clear()

    return np.mean(factor), np.std(factor)
