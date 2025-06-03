import ast
from typing import Callable



"""
    The heart of the fitting library, the Jacobian evaluation is designed to take a for loop function implementing a loss, 
    and then analyse and convert this to a function defining the jacobian of the cost function.

    It relies on all code being fundamentally expressed as jax based manipulations of input and output arrays.

"""

def jacobian( cost_function: Callable):
    tree = ast.parse(cost_function.__code__)

    # find every case where a value is written to the output array.

    # for each of these cases, trace the function to all cases where the input is derived from params.

    # evaluate the local derivatives of all permuatations over the code.

    # create local optimised multivariate chain rules to do this optimisation.

    # create this generic compute graph.

    # split the compute graph into chunks that are jax compilable

    # optimise and return the results.


