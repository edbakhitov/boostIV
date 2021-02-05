# ======================================================================================================================

# Auxiliary functions for boostIV

# ======================================================================================================================

import numpy as np
import numdifftools as nd
from patsy import dmatrix


# Use single-hidden-layer nn as a weak learner
def sigmoid_wl(x, gamma):
    # X is n by k + 1, including the intercept
    # gamma is k + 1 by 1
    t = x @ gamma
    # avoid overflow
    # sigmoid = np.exp(t - t.max()) / (np.exp(-t.max()) + np.exp(t - t.max()))
    sigmoid = 0.5 * (np.tanh(t) + 1)
    return sigmoid

# sigmoid jacobian
def sigmoid_wl_jac(x, gamma):
    obj = lambda z: sigmoid_wl(x, z)
    return nd.Jacobian(obj)(gamma)

# Use a cubic spline with 2 nodes as a weak learner
def cubic_spline(x, gamma, normalize=True):
    # X is n by k + 1, including the intercept
    # gamma is (3 + 2) * k + 1
    # 1. normalize the data
    X = x[:, 1:]
    if normalize == True:
        X = (X - X.min(0)) / (X.max(0) - X.min(0))
    # 2. calculate knots
    percentiles = [1/3, 2/3]
    knots = np.quantile(X, percentiles, axis=0)
    # 3. form basis functions
    basis = np.hstack(
        [dmatrix('bs(X, knots=' + '(' + ', '.join(str(z) for z in knots[:, j]) + ')' + ', degree=' + str(3) + ', include_intercept=False,'
                 + ' lower_bound=0, upper_bound=1)', {"X": X[:, j]}, return_type='dataframe').drop(['Intercept'], axis=1) for j in range(X.shape[1])])

    return np.c_[np.ones(X.shape[0]), basis] @ gamma

# cubic spline jacobian
def cubic_spline_jac(x, gamma):
    obj = lambda z: cubic_spline(x, z)
    return nd.Jacobian(obj)(gamma)

# L2-loss objective function
def fsam_ls_iv(params, r, x, iv, wl_model):
    beta = params[0]
    gamma = params[1:]
    # dimension check
    if len(iv.shape) == 1:
        iv = iv.reshape(len(iv), 1)
    # projection matrix Pz
    Pz = iv @ np.linalg.pinv(iv.T @ iv) @ iv.T
    wl = wl_model(x, gamma)
    return r - beta * Pz @ wl

# L2-loss objective function: not fitting the bs weight
def fsam_ls_iv_nw(params, r, x, iv, wl_model):
    # dimension check
    if len(iv.shape) == 1:
        iv = iv.reshape(len(iv), 1)
    # projection matrix Pz
    Pz = iv @ np.linalg.pinv(iv.T @ iv) @ iv.T
    wl = wl_model(x, params)
    return r - Pz @ wl

# Auxiliary function to split a matrix of parameters into matrices of parameters corresponding to different folds
def blockwise_view(a, blockshape, aslist=False, require_aligned_blocks=True):
    """
    Return a 2N-D view of the given N-D array, rearranged so each ND block (tile)
    of the original array is indexed by its block address using the first N
    indexes of the output array.
    Note: This function is nearly identical to ``skimage.util.view_as_blocks()``, except:
          - "imperfect" block shapes are permitted (via require_aligned_blocks=False)
          - only contiguous arrays are accepted.  (This function will NOT silently copy your array.)
            As a result, the return value is *always* a view of the input.
    Args:
        a: The ND array
        blockshape: The tile shape
        aslist: If True, return all blocks as a list of ND blocks
                instead of a 2D array indexed by ND block coordinate.
        require_aligned_blocks: If True, check to make sure no data is "left over"
                                in each row/column/etc. of the output view.
                                That is, the blockshape must divide evenly into the full array shape.
                                If False, "leftover" items that cannot be made into complete blocks
                                will be discarded from the output view.
    Here's a 2D example (this function also works for ND):
    >>> a = np.arange(1,21).reshape(4,5)
    >>> print(a)
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    >>> view = blockwise_view(a, (2,2), require_aligned_blocks=False)
    >>> print(view)
    [[[[ 1  2]
       [ 6  7]]
    <BLANKLINE>
      [[ 3  4]
       [ 8  9]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[11 12]
       [16 17]]
    <BLANKLINE>
      [[13 14]
       [18 19]]]]
    Inspired by the 2D example shown here: http://stackoverflow.com/a/8070716/162094
    """
    assert a.flags["C_CONTIGUOUS"], "This function relies on the memory layout of the array."
    blockshape = tuple(blockshape)
    outershape = tuple(np.array(a.shape) // blockshape)
    view_shape = outershape + blockshape

    if require_aligned_blocks:
        assert (
            np.mod(a.shape, blockshape) == 0
        ).all(), "blockshape {} must divide evenly into array shape {}".format(blockshape, a.shape)

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * np.array(blockshape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = np.lib.stride_tricks.as_strided(a, shape=view_shape, strides=(inter_block_strides + intra_block_strides))
    return view

