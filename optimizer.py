import os
import h5py
import shutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from model2optimize import model2optimize
from objective_function import objective_function
from h5_helpers import load_h5_to_dict, _save_dict_to_h5

# GA toolbox
from GA.ga_toolbox.population       import population
from GA.ga_toolbox.selection_uniq   import selection_uniq
from GA.ga_toolbox.crossover        import crossover
from GA.ga_toolbox.mutation         import mutation
from GA.ga_toolbox.mutationV        import mutationV
from GA.ga_toolbox.mutation_single  import mutation_single
from GA.ga_toolbox.fitness_function import fitness_function
from GA.ga_toolbox.gradient_search import gradient_search

# Gradient toolbox
from GA.gradient_toolbox.evaluation       import evaluation
from GA.gradient_toolbox.selection_best   import selection_best

# ==========================================================================
def _run_and_save(ref, result_path):
    """
    Inline translation of MATLAB run_ga.  Runs the full GA loop, saves the
    result as an HDF5 file, and returns the best parameter set.

    Parameters
    ----------
    ref         : dict   model configuration 
    result_path : str    full path to the .h5 output file

    Returns
    -------
    p_post : np.ndarray  [nParams,]  — parameters from the last generation
    """
    # ------------------------------------------------------------------
    # 0.  Hyperparameters 
    # ------------------------------------------------------------------
    op = -1          # -1: minimise,  1: maximise

    N1 = 60          # population size
    N2 = 100         # crossover: number of pairs
    N3 = 100         # mutation:  number of pairs
    tg = 1           # total generations

    # Gradient-search configuration
    conf = {
        'gLoop': 10,   # iterations per gradient search
        'gL':    -12,
        'gU':    12,
        'gTol':  0.01,
        'op':    op,
        'myfunc': objective_function,
    }
    conf['gT'] = abs(conf['gU'] - conf['gL']) + 1

    # ------------------------------------------------------------------
    # 1.  Boundaries
    # ------------------------------------------------------------------
    LR      = ref['model']['boundary'][:, 0]
    UR      = ref['model']['boundary'][:, 1]
    nParams = len(LR)

    # Expose boundary at top level (MEPmodel_bio may need it)
    ref['boundary'] = ref['model']['boundary']

    # conf also needs the bounds and the full ref for gradient_search
    conf['LR']     = LR
    conf['UR']     = UR
    conf['y_goal'] = ref

    # ------------------------------------------------------------------
    # 2.  Collect previous solutions to seed the population
    # ------------------------------------------------------------------
    solution_ini = np.empty((0, nParams))

    # 2a. Primary result file for this subject
    primary_path = result_path          # already the .h5 path for this subject
    if os.path.isfile(primary_path):
        print(f'{primary_path} found.')
        with h5py.File(primary_path, 'r') as f:
            tmp = load_h5_to_dict(f)
        solution_ini = np.vstack([solution_ini, np.atleast_2d(tmp['p_post'])])

    # 2c. Clip seeded solutions to parameter bounds
    if solution_ini.size > 0:
        for i in range(nParams):
            solution_ini[:, i] = np.clip(solution_ini[:, i], LR[i], UR[i])

    # ------------------------------------------------------------------
    # 3.  Initialisation
    # ------------------------------------------------------------------
    print('======== Initialization ========')
    P = population(N1, nParams, LR, UR)           # [N1 x nParams] random solutions
    if solution_ini.size > 0:
        P = np.vstack([P, solution_ini])           # append seeded solutions

    F, E, _ = evaluation(P, objective_function, ref)  # F: [nSolutions,]  E: [T x nSolutions]
    P, F, E = selection_best(P, F, E, N1, op)      # keep best N1
    E1 = E[:, 0]                                   # residual of current best

    print('done')
    print(f'Minimum cost: {F[0]}')
    print('================================')
    F_crit = F[0]

    # History accumulators  (row w holds generation w data)
    K          = []   # [[avg_cost, best_cost], …]
    KP         = []   # [best_params, …]
    KS         = []   # [best_cost, …]
    GA_counter = []   # [0/1 per generation]

    # ------------------------------------------------------------------
    # 4.  Online-plot setup
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    plt.ion()
    plt.show()

    # ------------------------------------------------------------------
    # 5.  Main GA loop
    # ------------------------------------------------------------------
    w = 0   # generation index (0-based, matching list append)
    j = 1   # generation counter for stopping (mirrors MATLAB j=1 before loop)

    while True:

        # ---- 5a. Gradient search on current best ----
        print('======= Gradient search ========')
        Para_E_grd, F_grd, E_grd = gradient_search(P[0:1, :], E1, conf, F_crit)
        # Replace best if gradient improved it
        if op * F_grd[0] > op * F[0]:
            P[0, :]  = Para_E_grd[0, :]
            F[0]     = F_grd[0]
            E[:, 0]  = E_grd[:, 0]
        print('done')

        # ---- 5b. Single-parameter mutation of current best ----
        print('======= single-parameter mutation ========')
        P_ = mutation_single(P[0:1, :].ravel(), LR, UR)   # [nParams x nParams]
        F_, E_, _ = evaluation(P_, objective_function, ref)
        print('done')

        # ---- 5c. Gradient search on each single-param mutant ----
        print('======= Gradient search ========')
        Para_E_grd = np.empty_like(P_)
        F_grd      = np.empty(len(F_))
        E_grd      = np.empty_like(E_)
        for i in range(len(F_)):
            #print(f'[{i+1}/{len(F_)}] cost: {F_[i]:.6f}')
            pg, fg, eg = gradient_search(P_[i:i+1, :], E_[:, i:i+1], conf, F_crit)
            Para_E_grd[i, :] = pg[0, :]
            F_grd[i]         = fg[0]
            E_grd[:, i]      = eg[:, 0]

        # Replace mutants where gradient improved them
        index = op * F_grd > op * F_
        index = index.ravel()

        P_[index, :]   = Para_E_grd[index, :]
        F_[index]      = F_grd[index]
        E_[:, index]   = E_grd[:, index]

        # Append mutants to population
        P = np.vstack([P, P_])                     # [(N1 + nParams) x nParams]
        F = np.hstack([F, F_])
        E = np.hstack([E, E_])
        print('done')

        # Track best-after-gradient for GA-effectiveness check
        _, F_show, _ = selection_best(P, F, E, 1, op)
        print(f'best after gradient: {F_show[0]}')

        # ---- 5d. GA operators ----
        print('GA search...')
        P_mutV     = mutationV(P[:N1, :], 0.1, 0.9, LR, UR)          # N1 solutions
        P_cross    = crossover(P, N2)                                   # 2*N2 solutions
        P_mut      = mutation(P, N3)                                    # 2*N3 solutions
        P_new      = np.vstack([P_mutV, P_cross, P_mut])

        # Evaluate only the newly generated solutions
        F_new, _, _ = evaluation(P_new, objective_function, ref)

        # Merge all solutions (mirrors MATLAB indexing: P(N1+nParams+1:end) is P_new)
        P = np.vstack([P, P_new])
        F = np.hstack([F, F_new])

        # Selection: keep N1 unique best solutions
        P, F = selection_uniq(P, F, N1, N1, op, LR, UR)

        # Re-evaluate residual of new best solution
        _, E1_arr, _ = evaluation(P[0:1, :], objective_function, ref)
        E1 = E1_arr[:, 0]
        print('done')

        # ---- 5e. Record history ----
        avg_cost = F.sum() / N1
        K.append([avg_cost, F[0]])
        KP.append(P[0, :].copy())
        KS.append(F[0])
        F_crit = F[0]

        print('========')
        print(f'current best Loss: {KS[w]}')
        print('========')

        gof = fitness_function(ref['y0'], E1)
        print('========')
        print(f'current best R2: {gof}')
        print('========')

        # GA-effectiveness flag
        if F_show[0] > F[0]:
            print('GA works')
            GA_counter.append(1)
        else:
            print("GA doesn't work")
            GA_counter.append(0)

        # ---- 5f. Online plot ----
        _, houtput = objective_function(KP[-1], ref)
        K_arr  = np.array(K)
        GA_arr = np.array(GA_counter)

        for ax in axes:
            ax.cla()

        axes[0].plot(K_arr[:, 1], 'b.')
        axes[0].plot(K_arr[:, 0], 'r.')
        axes[0].set_title('Blue - Best            Red - Average')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Loss function')
        axes[0].set_yscale('log')
        axes[0].grid(True)

        axes[1].plot(F, 'b.')
        axes[1].set_xlabel('Chromosomes')
        axes[1].set_ylabel('Loss function')
        axes[1].set_yscale('log')
        axes[1].grid(True)

        axes[2].plot(KP[-1], '-ko')
        axes[2].set_title('parameter')

        axes[3].plot(ref['y0'], 'k', linewidth=1.5)
        axes[3].plot(
            houtput['sim']['simMEP2'], 'r', linewidth=1.0
        )
        axes[3].set_title('target & best fit')

        axes[4].plot(GA_arr, 'b.')
        axes[4].set_xlabel('Generations')
        suc_rate = GA_arr.sum() / len(GA_arr) if len(GA_arr) else 0
        axes[4].set_title(
            f'0--not work, 1--work, total success rate: {suc_rate:.2f}'
        )

        plt.pause(0.01)
        fig.canvas.draw()

        # ---- 5g. Stopping criteria ----
        w += 1
        j += 1

        if j > tg:          # max generations reached
            break
        if KS[-1] < 0.01:   # good-enough fit
            break

    # ------------------------------------------------------------------
    # 6.  Extract best overall result
    #     MATLAB: p_post = KP(end,:)  — last generation's best
    #     (The overall minimum across all generations is also reported.)
    # ------------------------------------------------------------------
    KS_arr = np.array(KS)
    KP_arr = np.array(KP)

    if op == -1:
        best_idx = int(np.argmin(KS_arr))
        print(f'minimum: {KS_arr[best_idx]}')
    else:
        best_idx = int(np.argmax(KS_arr))
        print(f'maximum: {KS_arr[best_idx]}')

    # p_post follows MATLAB convention: last generation's best
    p_post = KP_arr[-1, :]

    # ------------------------------------------------------------------
    # 7.  Backup previous result (if any) then save
    # ------------------------------------------------------------------
    if os.path.isfile(result_path):
        timestamp   = datetime.now().strftime('%Y-%m%d-%H%M')
        backup_path = os.path.splitext(result_path)[0] + f'_backup-{timestamp}.h5'
        shutil.copyfile(result_path, backup_path)
        print(f'Previous result backed up to: {backup_path}')

    _, ref_final = model2optimize(p_post, ref)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with h5py.File(result_path, 'w') as f:
        f.create_dataset('p_post', data=p_post)
        f.create_dataset('KP',     data=KP_arr)
        f.create_dataset('KS',     data=KS_arr)
        f.create_dataset('P',      data=P)
        grp = f.create_group('ref')
        _save_dict_to_h5(grp, ref_final)

    print('fitted result saved:')
    print(result_path)

    return p_post