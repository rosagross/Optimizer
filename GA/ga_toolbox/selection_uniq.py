import numpy as np
from GA.ga_toolbox.population import population


def selection_uniq(P1, F, p, r, op, LR, UR):
    """
    Select p unique solutions, keeping the top-r best and randomly sampling
    the rest from the remaining pool.

    Parameters
    ----------
    P1 : np.ndarray  [n_pop, n_parameter]  population
    F  : np.ndarray  [n_pop,]               fitness values
    p  : int         desired output population size
    r  : int         number of top solutions guaranteed to be kept
    op : int         -1 → select minimum, +1 → select maximum
    LR : array-like  lower boundary
    UR : array-like  upper boundary

    Returns
    -------
    YY1 : np.ndarray  [p, n_parameter]  selected population
    YY2 : np.ndarray  [p,]               fitness of YY1
    """
    # Turn minimisation into maximisation if necessary
    F = op * F.copy()
    dim = P1.shape[1]

    # Remove inf entries
    index = np.isinf(F)
    F  = F[~index]
    P1 = P1[~index, :]

    # Remove nan entries
    index = np.isnan(F)
    F  = F[~index]
    P1 = P1[~index, :]

    # Keep only unique rows
    P1, ip = np.unique(P1, axis=0, return_index=True)
    F = F[ip]

    # Sort from high to low — best first
    index = np.argsort(F)[::-1]
    F  = F[index]
    P1 = P1[index, :]

    # Recheck length; pad with random solutions if fewer than p remain
    len_counter = len(F)
    if len_counter < p:
        new_len = p - len_counter
        P1 = np.vstack([P1, population(new_len, dim, LR, UR)])
        F  = np.concatenate([F, np.full(new_len, np.nan)])

    # Select best r, then randomly sample the rest
    if len_counter > p:
        P1_best = P1[:r, :]
        F_best  = F[:r]

        P2 = P1[r+1:, :]
        F2 = F[r+1:]

        index = np.random.permutation(len(F2))
        P2 = P2[index, :]
        F2 = F2[index]

        P2 = P2[:p - r, :]
        F2 = F2[:p - r]

        P1 = np.vstack([P1_best, P2])
        F  = np.concatenate([F_best, F2])

    YY1 = P1[:p, :]
    YY2 = op * F[:p]   # turn back to original sign

    return YY1, YY2