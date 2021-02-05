# ======================================================================================================================

# Functions to compute fitted predictions

# ======================================================================================================================

import numpy as np
import sys
from boostIV.primitives import sigmoid_wl, cubic_spline, blockwise_view


###################################################

### Get fit ###

###################################################

# get estimated basis functions
def get_bf_hat(Xmat, param_mat, wl_model, M, n_split):
    if wl_model == sigmoid_wl:
        n_param = 1 + Xmat.shape[1]
    elif wl_model == cubic_spline:
        n_param = ((3 + 2) * (Xmat.shape[1] - 1) + 1) + 1
    else:
        sys.exit('No weak learner type matched: pick sigmoid_wl or cubic_spline.')
    n_obs = Xmat.shape[0]
    bfs_hat_by_fold = np.empty((n_obs, M, n_split))
    # split the parameter matrix by fold
    # split the parameter matrix by fold: to maintain C-contiguity for blockwise_view function
    param_to_split = np.zeros((n_param * n_split, M))
    for m in range(M):
        param_to_split[:, m] = param_mat[:, m + 1]
    param_mat_by_fold = blockwise_view(param_to_split, (n_param, M))
    for l in range(n_split):
        for k in range(M):
            bfs_hat_by_fold[:, k, l] = wl_model(Xmat, param_mat_by_fold[l][0][1:n_param, k])
    bfs_hat = np.mean(bfs_hat_by_fold, axis=2)
    return bfs_hat


def get_boostIV_fit(Xmat, param_mat, wl_model, M, n_split):
    # number of parameters
    if wl_model == sigmoid_wl:
        n_param = 1 + Xmat.shape[1]
    elif wl_model == cubic_spline:
        n_param = ((3 + 2) * (Xmat.shape[1] - 1) + 1) + 1
    else:
        sys.exit('No weak learner type matched: pick sigmoid_wl or cubic_spline.')
    n_obs = Xmat.shape[0]
    avg_model = np.empty((n_obs, n_split))
    # split the parameter matrix by fold: to maintain C-contiguity for blockwise_view function
    param_to_split = np.zeros((n_param * n_split, M))
    for m in range(M):
        param_to_split[:, m] = param_mat[:, m + 1]
    param_mat_by_fold = blockwise_view(param_to_split, (n_param, M))  # first column is the intercept
    for l in range(n_split):
        bfs_fit = np.zeros((n_obs, M))
        for k in range(M):
            bfs_fit[:, k] = wl_model(Xmat, param_mat_by_fold[l][0][1:n_param, k])
        avg_model[:, l] = bfs_fit @ param_mat_by_fold[l][0][0, :]
    boost_fit = np.mean(avg_model, axis=1) + param_mat[:, 0].mean()
    return boost_fit


def get_post_boostIV_fit(param, wl_model, M, X, n_split, cf=True):
    """
    :param param: output of post_boostIV_crossfit
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param M: number of boosting iterations
    :param X: (out of sample) data to evaluate the structural function
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :return: h_fun estimate
    """
    # dimension check
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    # extract parameters
    bf_param = param['bf_param']
    model_fit = param['bf_weights']
    K = len(bf_param)
    h_post_boost = []
    for k in range(K):
        if cf:
            bfs_hat_k = get_bf_hat(X, bf_param[k], wl_model, M, n_split)
            h_post_boost_k = model_fit[k].predict(bfs_hat_k)
        else:
            h_post_boost_k = model_fit[k].predict(np.vstack([wl_model(X, bf_param[k][1:, j]) for j in range(M)]).T)
        h_post_boost.append(h_post_boost_k)
    h_post_boost = np.vstack(h_post_boost).mean(axis=0)
    return h_post_boost
