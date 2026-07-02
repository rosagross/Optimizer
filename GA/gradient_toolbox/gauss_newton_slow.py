import numpy as np
from GA.gradient_toolbox.NMM_diff_A_lfm import NMM_diff_A_lfm
from GA.gradient_toolbox.multi_lavenberg_regulization import multi_lavenberg_regulization
from GA.gradient_toolbox.evaluation import evaluation
from GA.gradient_toolbox.selection_best import selection_best


def gauss_newton_slow(op, Para_E_test, E_test, func, y_goal,
                      reg0, reg1, steps, loop, tol, LR, UR, fit_crit):
    """
    Iterative Gauss-Newton optimisation with Levenberg-Marquardt regularisation.

    Parameters
    ----------
    op           : int        -1 → minimise, +1 → maximise
    Para_E_test  : np.ndarray [nParams,]  initial parameters
    E_test       : np.ndarray [nData,]    initial residual
    func         : callable   objective: error, houtput = func(P, y_goal)
    y_goal       : target output
    reg0         : float      log10 of minimum regularisation (passed to MLR)
    reg1         : float      log10 of maximum regularisation (passed to MLR)
    steps        : int        number of regularisation candidates per iteration
    loop         : int        maximum number of iterations
    tol          : float      convergence tolerance (stop if improvement < tol)
    LR           : array-like lower boundary
    UR           : array-like upper boundary
    fit_crit     : not currently active (mirrors commented-out MATLAB code)

    Returns
    -------
    fit_after_g   : float      best fitness after gradient search
    Para_E_after_g: np.ndarray [nParams,]  best parameters after gradient search
    E_after_g     : np.ndarray [nData,]    residual of best solution
    """

    F_       = []           # fitness history
    Para_E_  = []           # parameter history  [iter x nParams]
    E_       = []           # residual history   [nData x iter]

    j = 0
    while j < loop:
        # ---- Jacobian ----
        J = NMM_diff_A_lfm(Para_E_test, E_test, func, y_goal)

        # ---- Candidate updates via Levenberg-Marquardt ----
        Para_E_new_group = multi_lavenberg_regulization(
            steps, reg0, reg1, Para_E_test, J, E_test, LR, UR
        )   # [steps x nParams]

        print(f'[{j + 1}/{loop}] ', end='', flush=True)

        # ---- Evaluate candidates ----
        F_grp, E_grp, _ = evaluation(Para_E_new_group, func, y_goal)

        # ---- Select best candidate ----
        Para_E_new, fit_new_arr, E_new = selection_best(
            Para_E_new_group, F_grp, E_grp, 1, op
        )
        fit_new   = float(fit_new_arr[0])
        E_test    = E_new          # [nData,]
        Para_E_test = Para_E_new       # [nParams,]

        print(fit_new, flush=True)

        F_.append(fit_new)
        Para_E_.append(Para_E_test.copy())
        E_.append(E_test.copy())

        j += 1

        # ---- Convergence check ----
        if len(F_) > 1 and op * F_[-1] - op * F_[-2] < tol:
            print(f'Quit: improvement < tol({tol:g})')
            break

    # ---- Return best overall or last iterate ----
    F_arr      = np.array(F_)             # [iter,]
    Para_E_mat = np.vstack(Para_E_)       # [iter x nParams]
    E_mat      = np.column_stack(E_)      # [nData x iter]

    if j == loop:
        # Loop ran to completion — pick the best across all iterations
        YY1, fit_after_arr, YY3 = selection_best(
            Para_E_mat, F_arr, E_mat, 1, op
        )
        Para_E_after_g = YY1
        fit_after_g    = float(fit_after_arr[0])
        E_after_g      = YY3
    else:
        # Converged early — use the last iterate
        Para_E_after_g = Para_E_test
        fit_after_g    = fit_new
        E_after_g      = E_test

    return fit_after_g, Para_E_after_g, E_after_g