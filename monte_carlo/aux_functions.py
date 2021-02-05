# ======================================================================================================================

# Auxiliary functions: NPIV, DeepIV and DeepGMM estimators

# ======================================================================================================================

import numpy as np
import sys
from sklearn.preprocessing import PolynomialFeatures
from econml.deepiv import DeepIVEstimator
import keras
from patsy import dmatrix
from DeepGMM.methods.DCM import DCM
from DeepGMM.methods.simple_method import simple_method
from DeepGMM.methods.toy_method import toy_method
import torch


##########################################################

                    ### NPIV ###

##########################################################

# spline basis
def spline_basis(x, degree, n_knots, normalize=True):
    """
    Spline basis functions
    :param x: input matrix
    :param degree: spline degree, must be greater than 2
    :param n_knots: number of interior knots (equally spaced)
    :param normalize: True by default -> to [0, 1]
    """
    # x is n by k (no intercept)
    # n_basis_fun is (degree + n_knots) * k + 1
    # degree must be greater than 2
    if degree < 2:
        raise ValueError('Spline degree should be greater or equal than 2.')
    # 1. dimension check + normalize the data
    if np.sum(x.shape) == 1:
        x = x.reshape(-1, 1)
    if normalize == True:
        X = (x - x.min(0)) / (x.max(0) - x.min(0))

    # 2. calculate knots
    percentiles = np.arange(1, n_knots + 1) / (n_knots + 1)
    knots = np.quantile(X, percentiles, axis=0)

    # 3. form basis functions
    basis = np.hstack(
        [dmatrix('bs(X, knots=' + '(' + ', '.join(str(z) for z in knots[:, j]) + ')' + ', degree=' + str(degree) + ', include_intercept=False,'
                 + ' lower_bound=0, upper_bound=1)', {"X": X[:, j]}, return_type='dataframe').drop(['Intercept'], axis=1) for j in range(X.shape[1])])

    return np.c_[np.ones(X.shape[0]), basis]


def npiv_fit(data, basis, degree):
    """
    :param data: a dictionary, which is a tuple (X, Y, Z)
    :param basis: type of basis expansion for NPIV: POLY or SPL
    :param degree: polynomial degree for basis expansion
    :return: sieve coefficients
    """
    # upload variables from the data
    X, Y, Z = data['X'], data['Y'], data['Z']
    # shape check: if X/Z are vectors => into matrices
    if len(X.shape) == 1:
        X = X.reshape(len(X), 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(len(Z), 1)
    # specify sieves: polynomial sieve includes interaction terms => more basis functions, while B-splines do not include interactions
    if basis == 'POLY':
        poly = PolynomialFeatures(degree)
        X_mat = poly.fit_transform(X)
        Z_mat = poly.fit_transform(Z)
    elif basis == 'SPL':
        knots_x = np.quantile(X, [0.25, 0.5, 0.75], axis=0)
        knots_z = np.quantile(Z, [0.25, 0.5, 0.75], axis=0)
        X_bs = np.hstack(
            [dmatrix('bs(X, knots=' + '(' + str(knots_x[0, j]) + ', ' + str(knots_x[1, j]) + ', ' + str(knots_x[2, j]) + ')' + ', degree=' + str(degree) +
                     ', include_intercept=False)', {"X": X[:, j]}, return_type='dataframe').drop(['Intercept'], axis=1) for j in range(X.shape[1])])
        X_mat = np.c_[np.ones(X.shape[0]), X_bs]
        Z_bs = np.hstack(
            [dmatrix('bs(Z, knots=' + '(' + str(knots_z[0, j]) + ', ' + str(knots_z[1, j]) + ', ' + str(knots_z[2, j]) + ')' + ', degree=' + str(degree) +
                     ', include_intercept=False)', {"Z": Z[:, j]}, return_type='dataframe').drop(['Intercept'], axis=1) for j in range(Z.shape[1])])
        Z_mat = np.c_[np.ones(Z.shape[0]), Z_bs]
    else:
        sys.exit('No basis type matched: pick either POLY or SPL')
    # estimation:
    """
    To avoid SVD convergence problems add some noise to the iv Gram matrix Z'Z: 'ridge'-regularize the matrix
    """
    zz = Z_mat.T @ Z_mat
    while True:
        try:
            pz = Z_mat @ np.linalg.pinv(zz) @ Z_mat.T
            break
        except np.linalg.LinAlgError:
            print('Pseudoinverse cannot be calculated: SVD does not converge. Add noise to Z.T @ Z and try again')
        # update the seed and the counter
        zz += 1e-10 * np.eye(Z_mat.shape[1])

    beta_npiv = np.linalg.inv(X_mat.T @ pz @ X_mat) @ X_mat.T @ pz @ Y
    return beta_npiv


def get_npiv_fit(X_oos, beta_npiv, basis, degree):
    """
    :param X_oos: regressors for predictions
    :param beta_npiv: estimated sieve coefficients
    :param basis: type of basis expansion for NPIV: POLY or SPL
    :param degree: polynomial degree for basis expansion
    :return: fitted values
    """
    # prediction at a point
    if len(X_oos.shape) == 1:
        if basis == 'POLY':
            poly = PolynomialFeatures(degree)
            X_poly_oos = poly.fit_transform(X_oos.reshape(1, len(X_oos)))
        elif basis == 'SPL':
            # knots_x = np.quantile(data['X'], [0.25, 0.5, 0.75], axis=0)
            X_bs = np.hstack(
                [dmatrix('bs(X, knots=(1,1,1), degree=' + str(degree) +
                         ', include_intercept=False)', {"X": X_oos[j]}, return_type='dataframe').drop(['Intercept'], axis=1) for j in range(len(X_oos))])
            X_poly_oos = np.append(1, X_bs)
        else:
            sys.exit('No basis type matched: pick either POLY or SPL')
        h_npiv_oos = X_poly_oos @ beta_npiv
    # prediction for a dataset
    elif len(X_oos.shape) == 2:
        if basis == 'POLY':
            poly = PolynomialFeatures(degree)
            X_poly_oos = poly.fit_transform(X_oos)
        elif basis == 'SPL':
            knots_x = np.quantile(X_oos, [0.25, 0.5, 0.75], axis=0)
            X_bs = np.hstack(
                [dmatrix('bs(X, knots=' + '(' + str(knots_x[0, j]) + ', ' + str(knots_x[1, j]) + ', ' + str(knots_x[2, j]) + ')' + ', degree=' + str(degree) +
                         ', include_intercept=False)', {"X": X_oos[:, j]}, return_type='dataframe').drop(['Intercept'], axis=1) for j in range(X_oos.shape[1])])
            X_poly_oos = np.c_[np.ones(X_bs.shape[0]), X_bs]
        else:
            sys.exit('No basis type matched: pick either POLY or SPL')
        h_npiv_oos = X_poly_oos @ beta_npiv
    else:
        sys.exit('Dimension mismatch: X_oos should be either a vector or a matrix.')
    return h_npiv_oos

##########################################################

                    ### Deep IV ###

##########################################################
def deepiv_fit(data):
    """
    Add description
    :param data: a dictionary, which is a tuple (X, Y, Z)
    :return: an estimator object
    """
    # upload variables from the data
    X, Y, Z = data['X'], data['Y'], data['Z']
    # shape check: if X/Z are vectors => into matrices
    if len(X.shape) == 1:
        X = X.reshape(len(X), 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(len(Z), 1)
    # specify the treatment model
    treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(Z.shape[1] + 1,)),
                                        keras.layers.Dropout(0.17),
                                        keras.layers.Dense(64, activation='relu'),
                                        keras.layers.Dropout(0.17),
                                        keras.layers.Dense(32, activation='relu'),
                                        keras.layers.Dropout(0.17)])
    # specify the response model
    response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1] + 1,)),
                                       keras.layers.Dropout(0.17),
                                       keras.layers.Dense(64, activation='relu'),
                                       keras.layers.Dropout(0.17),
                                       keras.layers.Dense(32, activation='relu'),
                                       keras.layers.Dropout(0.17),
                                       keras.layers.Dense(1)])
    keras_fit_options = {"epochs": 30,
                         "validation_split": 0.1,
                         "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]}

    deepIvEst = DeepIVEstimator(n_components=10,  # number of gaussians in our mixture density network
                                m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),  # treatment model
                                h=lambda t, x: response_model(keras.layers.concatenate([t, x])),  # response model
                                n_samples=1,  # number of samples to use to estimate the response
                                use_upper_bound_loss=False,  # whether to use an approximation to the true loss
                                n_gradient_samples=1,
                                # number of samples to use in second estimate of the response (to make loss estimate unbiased)
                                optimizer='adam',
                                # Keras optimizer to use for training - see https://keras.io/optimizers/
                                first_stage_options=keras_fit_options,  # options for training treatment model
                                second_stage_options=keras_fit_options)  # options for training response model
    deepIvEst.fit(Y=Y, T=X, X=np.ones((X.shape[0], 1)), Z=Z)
    return deepIvEst


##########################################################

                    ### Deep GMM ###

##########################################################

def to_tensor(data):

    # extract data
    X, Y, Z, g = data['X'], data['Y'], data['Z'], data['g']
    # convert into tensors
    x = torch.as_tensor(X).double()
    y = torch.as_tensor(Y).double()
    z = torch.as_tensor(Z).double()
    g = torch.as_tensor(g).double()
    return {'X': x, 'Y': y, 'Z': z, 'g': g}


def deepGMM_fit(data, model='dcm'):
    """
    Add description
    :param data: a dictionary conataining train, dev, and test data
    :return: an estimator object
    """

    ### 1. add torch tensors to data ###
    data_train, data_dev, data_test = to_tensor(data['train']), to_tensor(data['dev']), to_tensor(data['test'])

    ### 2. specify estimator's dimensions ###
    if len(data_train['X'].shape) == 1:
        g_dim = 1
    else:
        g_dim = data_train['X'].shape[1]

    if len(data_train['Z'].shape) == 1:
        f_dim = 1
    else:
        f_dim = data_train['Z'].shape[1]

    ### 3. extract data ###
    x_train, z_train, y_train = data_train['X'], data_train['Z'], data_train['Y']
    x_dev, z_dev, y_dev, g_dev = data_dev['X'], data_dev['Z'], data_dev['Y'], data_dev['g']
    x_test, g_test = data_test['X'], data_test['g']

    # run DeepGMM
    if model == 'dcm':
        method = DCM(input_dim_g=g_dim, input_dim_f=f_dim, enable_cuda=False)
    elif model == 'simple':
        method = simple_method(input_dim_g=g_dim, input_dim_f=f_dim, enable_cuda=False)
    elif model == 'toy':
        method = toy_method(input_dim_g=g_dim, input_dim_f=f_dim, enable_cuda=False)
    else:
        raise ValueError('No method matched: pick dcm or simple')
    method.fit(x_train, z_train, y_train, x_dev, z_dev, y_dev, g_dev=g_dev, verbose=True)
    g_pred_test = method.predict(x_test)

    return g_pred_test



