import numpy as np
from GA.gradient_toolbox.gradient_repair import gradient_repair


def multi_lavenberg_regulization(n, reg0, reg1, Para_E, J, E, LR, UR):
    """
    Generate n candidate parameter updates via Levenberg-Marquardt regularisation,
    each with a different regularisation strength on a log scale [reg0, reg1].

    Parameters
    ----------
    n       : int        number of candidate solutions to return
    reg0    : float      log10 of minimum regularisation value
    reg1    : float      log10 of maximum regularisation value
    Para_E  : np.ndarray [nParams,]  current parameter vector
    J       : np.ndarray [nData, nParams]  Jacobian matrix
    E       : np.ndarray [nData,]  current residual (f(x) - y_goal)
    LR      : float or array-like  lower boundary
    UR      : float or array-like  upper boundary

    Returns
    -------
    Y : np.ndarray  [n x nParams]  updated candidate solutions
    """

    nParams = len(Para_E)

    Y = np.zeros((n, nParams))

    reg = 10.0 ** np.linspace(reg0, reg1, n)

    for i in range(n):
        # Levenberg-Marquardt regularisation step
        try:
            D = np.linalg.pinv(J.T @ J + reg[i] * np.eye(nParams))
        except Exception:
            return Y

        d = -D @ J.T @ E   # parameter update (f(x) - y_goal convention)

        if np.any(np.isnan(d)):
            # equivalent to MATLAB's keyboard — raise to allow inspection
            raise RuntimeError(
                f'NaN detected in gradient update at regularisation step {i}. '
                f'reg={reg[i]:.4g}, check J and E.'
            )

        Para_E_new = Para_E + d

        # Clamp to boundary
        Y[i, :] = gradient_repair(Para_E_new, LR, UR)
    np.savetxt("Y.txt", Y)
    return Y