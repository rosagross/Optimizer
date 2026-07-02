import numpy as np


def selection_best(P, F_, E, p, op):
    """
    Select the top-p solutions by fitness.

    Parameters
    ----------
    P  : np.ndarray  [n_pop, n_parameter]  population
    F_ : np.ndarray  [n_pop,]               fitness values
    E  : np.ndarray  [n_data_sample, n_pop] residuals (y - h_output)
    p  : int         number of solutions to return
    op : int         -1 → select minimum fitness, +1 → select maximum fitness

    Returns
    -------
    YY1 : np.ndarray  [p, n_parameter]      selected population
    YY2 : np.ndarray  [p,]                  fitness of YY1
    YY3 : np.ndarray  [n_data_sample, p]    residuals of YY1
    """
    # Turn minimisation into maximisation if necessary
    F = F_.copy()
    F = op * F
    # Sort from high to low — best first
    index = np.argsort(F)[::-1]
    F = np.sort(F)[::-1]
    P = P[index, :]
    E = E[:, index]


    YY1 = P[:p, :]
    YY2 = op * F[:p]   # turn back to original sign
    YY3 = E[:, :p]

    if p == 1:
        YY1 = YY1.ravel()
        YY3 = YY3.ravel()

    return YY1, YY2, YY3