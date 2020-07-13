from ..utils import Falsify


def CorrectionFactor(p, number_of_runs, method, X, y):

    factor = []
    for _ in range(number_of_runs):

        # Artificially falsify
        y_f = Falsify(y, p)

        # Correct labels
        y_corrected = method.fit_transform(X, y_f)

        N = X.shape[0]
        factor.append(((y == y_corrected).sum() - (1-p)*N)/(p*N))

    factor = np.array(factor)
    return np.mean(factor), np.std(factor)
