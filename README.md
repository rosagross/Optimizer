# Optimizer

This repository contains a hybrid optimization algorithm that combines a genetic algorithm (GA) with local gradient-based search to fit a model's parameters to a target dataset. The main entry point of the optimization pipeline is optimizer.py. The algorithm is adapted from the previous work of Dr. Peng Wang [1,2] and his colleague, Dr. Vincent Chien [2]. The MATLAB implementation can be found in https://github.com/vscChien/MEPmodeling.

## Overview

The optimizer (`_run_and_save` in `optimizer.py`) runs an iterative loop that:

1. Initializes a random population of candidate parameter sets within given
   boundaries (optionally seeded with a previous solution, if a result file
   already exists for the subject/case being optimized).
2. Evaluates the population using a user-supplied `objective_function`.
3. Refines the current best solution with a local `gradient_search`.
4. Applies single-parameter mutation followed by another round of gradient
   search to the mutants.
5. Generates new candidate solutions via GA operators — variance-based
   mutation (`mutationV`), `crossover`, and `mutation` — and evaluates them.
6. Selects the best unique solutions to form the next generation
   (`selection_uniq`).
7. Tracks the best cost, best parameters, and GA effectiveness across
   generations, and updates a live diagnostic plot.
8. Repeats until a maximum number of generations is reached or the fit is
   good enough (best cost below a threshold).
9. Saves the final result — including the best parameters, full history, and
   final population — to an HDF5 file, backing up any pre-existing result
   file first.

## Repository Structure

```
optimizer.py                  # Main optimization loop (entry point)
model2optimize.py             # USER-IMPLEMENTED — model prediction function
objective_function.py         # USER-IMPLEMENTED — residual/objective function
h5_helpers.py                 # HDF5 read/write helpers
GA/
  ga_toolbox/
    population.py
    selection_uniq.py
    crossover.py
    mutation.py
    mutationV.py
    mutation_single.py
    fitness_function.py
    gradient_search.py
  gradient_toolbox/
    evaluation.py
    selection_best.py
```

## Required User Implementations

To use this optimizer with your own model, you must implement two functions:
`model2optimize` and `objective_function`. These are the only
model-specific pieces of the pipeline — everything else in the GA and
gradient-search toolboxes is generic.

### `model2optimize(p, ref)`

Runs your model forward with a given parameter set and returns its
predictions.

**Inputs**
- `p` — the parameter vector from the last optimization step (`p_post`),
  i.e. the candidate solution currently being evaluated.
- `ref` — a dictionary containing the model configuration (boundaries,
  target data, and any other information your model needs to run).

**Returns**
- The model's predicted output (e.g. a simulated signal or curve, in
  whatever structure your `objective_function` expects to consume).
- The updated `ref` dictionary, with the model configuration reflecting the
  result of this run (e.g. updated simulation outputs stored under `ref`
  for later use, such as plotting).

```python
def model2optimize(p, ref):
    """
    Parameters
    ----------
    p   : array-like, shape [nParams,]
        Parameter values to run the model with.
    ref : dict
        Model configuration dictionary.

    Returns
    -------
    predicted : array-like
        Model output/prediction generated using parameters p.
    ref : dict
        Updated model configuration dictionary.
    """
    ...
    return predicted, ref
```

### `objective_function(p, ref)`

Computes the residual vector between the model prediction and the target
data, for use by the GA and gradient-search routines.

**Inputs**
- `p` — the parameter vector from the last optimization step.
- `ref` — the model configuration dictionary.

**Behavior**
- Calls `model2optimize(p, ref)` to obtain the model prediction and the
  updated `ref`.
- Computes the difference between the predicted values and the target
  values (e.g. `predicted - target`).

**Returns**
- A vector containing the difference between the predicted values and the
  target values (the residual vector minimized/maximized by the optimizer).
- The updated `ref` dictionary returned by `model2optimize`.

```python
def objective_function(p, ref):
    """
    Parameters
    ----------
    p   : array-like, shape [nParams,]
        Parameter values to evaluate.
    ref : dict
        Model configuration dictionary.

    Returns
    -------
    residual : array-like
        Difference between predicted values and target values.
    ref : dict
        Updated model configuration dictionary (as returned by
        model2optimize).
    """
    predicted, ref = model2optimize(p, ref)
    residual = predicted - ref['y0']  # target values, e.g. stored in ref['y0']
    return residual, ref
```

> Note: `optimizer.py` expects `ref['model']['boundary']` (an `[nParams, 2]`
> array of lower/upper bounds per parameter) and `ref['y0']` (the target
> data used for the goodness-of-fit calculation and diagnostic plotting) to
> be present in the `ref` dictionary you pass in.

## Configuration

Key hyperparameters are set at the top of `_run_and_save`:

| Parameter | Description |
|---|---|
| `op` | Optimization direction: `-1` to minimize, `1` to maximize |
| `N1` | Population size |
| `N2` | Number of crossover pairs |
| `N3` | Number of mutation pairs |
| `tg` | Total number of generations |
| `conf['gLoop']` | Iterations per gradient search |
| `conf['gL']`, `conf['gU']` | Lower/upper gradient-search step bounds |
| `conf['gTol']` | Gradient search tolerance |

The loop also stops early if the best cost drops below `0.01`.

## Output

Results are saved to an HDF5 file at `result_path`, containing:

- `p_post` — best parameter set (from the final generation)
- `KP` — best parameters per generation
- `KS` — best cost per generation
- `P` — final population
- `ref` — the final model configuration dictionary (as returned by
  `model2optimize` for the best parameters)

If a result file already exists at `result_path`, it is backed up with a
timestamped filename before being overwritten, and its saved `p_post` is
used to seed the initial population.

## Live Diagnostics

While running, the optimizer displays a live-updating figure with:

1. Best and average cost per generation
2. Cost across all chromosomes in the current population
3. Current best parameter vector
4. Target data vs. the current best model fit
5. Whether the GA step improved on the gradient-search result, and the
   overall GA success rate
   
## Refernces
1. Wang, P., Kong, R., Kong, X., Liégeois, R., Orban, C., Deco, G., ... & Thomas Yeo, B. T. (2019). Inversion of a large-scale circuit model reveals a cortical hierarchy in the dynamic resting human brain. Science advances, 5(1), eaat7854.
2. Chien, V. S., Wang, P., Maess, B., Fishman, Y., & Knösche, T. R. (2023). Laminar neural dynamics of auditory evoked responses: Computational modeling of local field potentials in auditory cortex of non-human primates. NeuroImage, 281, 120364.
