import numpy as np


def fitness_function(y_goal, h_output):
    """
    Compute the goodness of fit: var(y - h_output) / var(y)  [i.e. 1 - R²]

    Parameters
    ----------
    y_goal   : np.ndarray  target output
    h_output : np.ndarray  model output

    Returns
    -------
    fit : float  fitness value (lower is better)
    """
    tmp = y_goal - h_output
    fit = np.var(tmp) / np.var(y_goal)  # 1 - R2

    return fit
