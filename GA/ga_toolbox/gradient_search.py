import numpy as np
from GA.gradient_toolbox.gauss_newton_slow import gauss_newton_slow


def gradient_search(P, E, conf, stop_crit):
    """
    Run gradient search on the solutions.

    Parameters
    ----------
    P         : np.ndarray  [N x nParams]  solutions
    E         : np.ndarray  [timepoints x N]  residual of solutions
    conf      : dict        configuration dictionary with keys:
                    'op', 'myfunc', 'y_goal', 'gL', 'gU', 'gT',
                    'gLoop', 'gTol', 'LR', 'UR'
    stop_crit : stop criterion for gradient search

    Returns
    -------
    P_post    : np.ndarray  [N x nParams]  solutions after gradient search
    F_post    : np.ndarray  [N,]           fitness (= sumsqr(error)) after gradient search
    E_post    : np.ndarray  [timepoints x N]  residual after gradient search
    """
    
    N, nParams = P.shape
    timepoints = E.shape[0]
    E = E.reshape(-1, 1)

    F_post = np.zeros(N)
    P_post = np.zeros((N, nParams))
    E_post = np.zeros((timepoints, N))

    for i in range(N):
        print(f'G{i}:')
        F_post[i], P_post[i, :], E_post[:, i] = gauss_newton_slow(
            conf['op'],
            P[i, :],  
            E[:, i],
            conf['myfunc'],
            conf['y_goal'],
            conf['gL'],
            conf['gU'],
            conf['gT'],
            conf['gLoop'],
            conf['gTol'],
            conf['LR'],
            conf['UR'],
            stop_crit,
        )

    return P_post, F_post, E_post