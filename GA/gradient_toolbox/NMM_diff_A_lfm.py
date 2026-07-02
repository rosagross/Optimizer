import numpy as np


def NMM_diff_A_lfm(para, E, myfunc, y_goal):
    """
    Estimate the Jacobian of a black-box function using forward finite differences.

    Approximation:
        df/dp_i ≈ ( f(p + h·eᵢ) − f(p) ) / h

    Parameters
    ----------
    para    : np.ndarray  [nParams,] current parameter vector
    E       : np.ndarray  [T,]  residual (error) at `para` (reference)
    myfunc  : callable    black-box function: error = myfunc(para, y_goal)
    y_goal  : target output passed through to myfunc

    Returns
    -------
    j : np.ndarray  [T x nParams]  Jacobian matrix
    """
    h = 1e-6

    para1 = para + h   # perturbed parameter vector (all elements shifted; only one used at a time)

    para_save = para.copy()
    T = len(E)
    nParams = len(para)

    f = np.zeros((T, nParams))

    for i in range(nParams):
        para_1 = para_save.copy()
        para_1[i] = para1[i]                         # perturb only parameter i

        E_new, _ = myfunc(para_1, y_goal)

        f[:, i] = (E_new - E) / h

    j = f   # [T x nParams]

    j[np.isnan(j)] = 0.0
    j[np.isinf(j)] = 0.0

    return j