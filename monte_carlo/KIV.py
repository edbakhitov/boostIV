# ======================================================================================================================

# KIV auxiliary functions: based on matlab codes of the authors
# https://github.com/r4hu1-5in9h/KIV

# ======================================================================================================================

import numpy as np
import os
from scipy import optimize

def make_psd(A):
    """ for numerical stability, add a small ridge to a symmetric matrix """
    # shape check: A should be a square matrix
    if A.shape[0] != A.shape[1]:
        raise TypeError('input matrix should be a square matrix')

    eps = 1e-10
    N = A.shape[0]
    A_psd = (A + A.T) / 2 + eps * np.eye(N)
    return A_psd


def data_split(X, Y, Z, frac):
    """ splits the data in two parts according to a fraction """
    # shape check: if X/Z is a vector => convert into a matrix
    if len(X.shape) == 1:
        X = X.reshape(len(X), 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(len(Z), 1)
    # splitting
    N = len(Y)
    n = int(np.round(frac * N))
    X1, X2 = X[0:n, :], X[n:N, :]
    Z1, Z2 = Z[0:n, :], Z[n:N, :]
    Y1, Y2 = Y[0:n], Y[n:N]
    # output
    df = {'X1': X1, 'X2': X2, 'Z1': Z1, 'Z2': Z2, 'Y1': Y1, 'Y2': Y2}
    return df

def med_inter(X):
    """
    :param X: input vector
    :return: median interpoint distance to use as the bandwidth
    """
    n_x = len(X)
    A = np.repeat(X.reshape(n_x, 1), n_x, axis=1)
    dist = np.abs(A - A.T).reshape(-1)
    v = np.median(dist)
    return v

def get_Kmat(X, Y, v):
    """
    returns the covariance matrix for the noiseless GP with RBF kernel at inputs X and Y
    :param X, Y: vectors of dim n_x and n_y
    :param v: bandwidth
    """
    n_x = len(X)
    n_y = len(Y)
    K_true = np.empty((n_x, n_y))

    # fill in the matrix
    for i in range(n_x):
        for j in range(n_y):
            K_true[i, j] = np.exp(-np.sum((X[i] - Y[j]) ** 2) / (2 * (v ** 2)))

    return K_true

def get_Kmat_mult(X, Y, v_vec):
    """
    calculates a multivariate RBF kernel as a product of scalar products of each column of X
    :param X and Y: matrices
    :param v_vec: vector of bandwidths
    """
    # shape check: if X/Y is a vector => convert into a matrix
    if len(X.shape) == 1:
        X = X.reshape(len(X), 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(len(Y), 1)
    # shape check: the number of columns should be the same
    if X.shape[1] != Y.shape[1]:
        raise TypeError('number of columns of input matrices must coincide')
    n_x = X.shape[0]
    n_y = Y.shape[0]
    d = X.shape[1]

    # calculate the kernel
    K_true = np.ones((n_x, n_y))
    for j in range(d):
        K_j = get_Kmat(X[:, j], Y[:, j], v_vec[j])
        K_true = np.multiply(K_true, K_j)

    return K_true

def get_K(X, Z, Y, X_test):
    """
    Precalculates kernel matrices for the 1st and 2nd stages
    :param X: endogenous regressors
    :param Z: IVs
    :param Y: response variable
    :param X_test: test sample
    :return: data dictionary
    """
    # shape check: if X/Z is a vector => convert into a matrix
    if len(X.shape) == 1:
        X = X.reshape(len(X), 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(len(Z), 1)
    # shape check: if oos_type is point, then X_test is d_x by 1 a vector => into [1, d_x] matrix
    if len(X_test.shape) == 1:
        X_test = X_test.reshape(1, len(X_test))

    # bandwidths
    v_x = np.array([med_inter(X[:, j]) for j in range(X.shape[1])])
    v_z = np.array([med_inter(Z[:, j]) for j in range(Z.shape[1])])

    # split the data
    df = data_split(X, Y, Z, frac=0.5)

    # calculate kernels
    K_XX = get_Kmat_mult(df['X1'], df['X1'], v_x)
    K_xx = get_Kmat_mult(df['X2'], df['X2'], v_x)
    K_xX = get_Kmat_mult(df['X2'], df['X1'], v_x)
    K_Xtest = get_Kmat_mult(df['X1'], X_test, v_x)
    K_ZZ = get_Kmat_mult(df['Z1'], df['Z1'], v_z)
    K_Zz = get_Kmat_mult(df['Z1'], df['Z2'], v_z)

    # output
    df_out = {'K_XX': K_XX, 'K_xx': K_xx, 'K_xX': K_xX, 'K_Xtest': K_Xtest,
              'K_ZZ': K_ZZ, 'K_Zz': K_Zz, 'Y1': df['Y1'], 'Y2': df['Y2']}
    return df_out

def KIV_pred(df, hyp, stage):
    """
    :param df: data frame produced by get_K
    :param hyp: hyperparameters
    :param stage: stage=(2,3) corresponds to stage 2 and testing
    :return: predictive mean for KIV
    """
    n = len(df['Y1'])
    m = len(df['Y2'])

    lam = hyp[0]
    xi = hyp[1]

    brac = make_psd(df['K_ZZ']) + lam * np.eye(n) * n
    W = df['K_XX'] @ np.linalg.inv(brac) @ df['K_Zz']
    brac2 = make_psd(W @ W.T) + m * xi * make_psd(df['K_XX'])
    alpha = np.linalg.inv(brac2) @ W @ df['Y2']

    if stage == 2:
        K_Xtest = df['K_XX']
    elif stage == 3:
        K_Xtest = df['K_Xtest']
    else:
        os.exit('stage should be equal to either 2 or 3')

    y_pred = (alpha.T @ K_Xtest).flatten()
    return y_pred

def KIV1_loss(df, lam):
    """
    :param df: data frame produced by get_K
    :param lam: 1st stage hyperparameter
    :return: 1st stage error of KIV
    """
    n = len(df['Y1'])
    m = len(df['Y2'])

    brac = make_psd(df['K_ZZ']) + lam * np.eye(n) * n
    gamma = np.linalg.inv(brac) @ df['K_Zz']
    loss = np.trace(df['K_xx'] - 2 * df['K_xX'] @ gamma + gamma.T @ df['K_XX'] @ gamma) / m
    return loss

def KIV2_loss(df, hyp):
    """
    :param df: data frame produced by get_K
    :param hyp: hyperparameters
    :return: 2nd stage error of KIV
    """
    n = len(df['Y1'])
    Y1_pred = KIV_pred(df, hyp, 2)
    loss = np.sum((df['Y1'] - Y1_pred) ** 2) / n
    return loss

def get_KIV(data, X_test):
    """
    This function estimates the model using KIV and provides out of sample estimates
    :param data: a dictionary, which is a tuple (X, Y, Z)
    :param X_test: out of sample data
    :return: out of sample estimates
    """
    X, Y, Z = data['X'], data['Y'], data['Z']
    # 1. calculate kernels
    df = get_K(X, Z, Y, X_test)

    # 2. initialize hyperparameters for tuning
    lam_0 = np.log(0.05)
    xi_0 = np.log(0.05)

    # 3. 1st stage tuning
    KIV1_obj = lambda lam: KIV1_loss(df, np.exp(lam))
    lam_star = optimize.fmin(KIV1_obj, lam_0)

    # 4. 2nd stage tuning
    KIV2_obj = lambda xi: KIV2_loss(df, [np.exp(lam_star), np.exp(xi)])
    xi_star = optimize.fmin(KIV2_obj, xi_0)

    # 5. evaluate out of sample using tuned hyperparameters
    Y_oos = KIV_pred(df, [np.exp(lam_star), np.exp(xi_star)], stage=3)

    return Y_oos

