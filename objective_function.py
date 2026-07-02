"""
objective_function.py  (USER-IMPLEMENTED for optimizer.py)
----------------------------------------------------------
Residual between the model prediction and the target data. The optimizer squares
and sums this vector to get the scalar cost (GA/gradient_toolbox/evaluation.py),
and uses the vector itself to build the Gauss-Newton Jacobian, so it must be a
1-D array of the *same fixed length* on every call.
"""

from model2optimize import model2optimize


def objective_function(p, ref):
    """Return (residual_vector, ref) where residual = predicted - ref['y0']."""
    predicted, ref = model2optimize(p, ref)
    residual = predicted - ref["y0"]
    return residual, ref
