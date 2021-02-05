# ======================================================================================================================

# Auxiliary functions generating data for the RCLM

# ======================================================================================================================

import numpy as np


def get_ccp(u_choice, T, list_J, N):
    ccp = np.zeros((sum(list_J), N))
    for t in range(T):
        u_choice_t = u_choice[sum(list_J[:t]): sum(list_J[:t + 1]), :]
        ccp[sum(list_J[:t]): sum(list_J[:t + 1]), :] = np.exp(u_choice_t) / (1 + np.exp(u_choice_t).sum(0).reshape(1, N))
    return ccp

""" Generate Data """
def gen_data(seed, spec):

    # DGP parameters
    distr = spec['distr']
    sim_design = spec['design']
    T = spec['T']
    J = spec['J']
    K_1 = spec['K_1']
    K_2 = spec['K_2']
    K_w = spec['K_w']
    N = spec['N']
    beta_true = spec['beta_true']
    price_rc = spec['price_rc']

    # set the seed
    np.random.seed(seed)

    # generate market indices
    list_J = T * [J]
    ids_market = np.hstack([[t] * list_J[t] for t in range(len(list_J))])

    '''-------------------------------------------------------------------------'''
    ''' 1. Generate raw characteristics and other variables						'''
    '''-------------------------------------------------------------------------'''

    # Non-truncated, normalized design
    if sim_design == 'Normal':
        x_1 = np.random.normal(loc=0, scale=1, size=(sum(list_J), K_1))
        x_2 = np.random.normal(loc=0, scale=1, size=(sum(list_J), K_2))
        w = np.random.normal(0, 1, size=(sum(list_J), K_w))
        xi = np.random.normal(loc=1, scale=0.15, size=(sum(list_J), 1))
        price = 2 * np.abs((x_2.sum(1).reshape(sum(list_J), 1) + w)) + xi
        delta = (np.c_[price, x_1, x_2] @ beta_true)[:, np.newaxis] + xi
    elif sim_design == 'Uniform':
        # Truncated design
        x_1 = np.random.uniform(1, 5, size=(sum(list_J), K_1))
        x_2 = np.random.uniform(1, 5, size=(sum(list_J), K_2))
        w = np.random.uniform(0, 1, size=(sum(list_J), K_w))
        xi = np.random.normal(1, 0.25, size=(sum(list_J), 1))
        price = x_2.sum(1).reshape(sum(list_J), 1) / 10 + xi + w
        delta = (np.c_[price, x_1, x_2] @ beta_true)[:, np.newaxis] + xi
    else:
        raise TypeError("No simulation design type matched.")

    '''-------------------------------------------------------------------------'''
    ''' 2. True features								                        '''
    '''-------------------------------------------------------------------------'''

    if price_rc:
        x_rc = np.c_[price, x_2]
    else:
        x_rc = x_2
    K_rc = x_rc.shape[1]
    v = np.zeros((T, K_rc * N))

    '''-------------------------------------------------------------------------'''
    ''' 3. RC distribution								                        '''
    '''-------------------------------------------------------------------------'''

    # Generate random coefficients
    if distr == "indep-normal":
        v_t = np.random.normal(loc=0, scale=0.25, size=(N, K_rc)).reshape(1, K_rc * N)
        for t in range(T):
            v[t, :] = v_t
            # draw different consumers across markets
            # v[t, :] = (np.random.normal(loc=0, scale=1, size=(N, K_rc)) @ cholesky(covx)).T.reshape(1, K_rc * N)
    elif distr == "corr-normal":
        covx = 0.3 * np.eye(K_rc) + 0.2
        v_t = (np.random.normal(loc=0, scale=1, size=(N, K_rc)) @ np.linalg.cholesky(covx)).T.reshape(1, K_rc * N)
        for t in range(T):
            v[t, :] = v_t
            # draw different consumers across markets
            # v[t, :] = (np.random.normal(loc = 0, scale = 1, size = (N, K_rc)) @ cholesky(covx)).T.reshape(1, K_rc * N)
    else:
        raise TypeError("No RC distribution type matched.")
    v = np.vstack([v[t:t+1, :].repeat(list_J[t], axis=0) for t in range(T)])

    '''-------------------------------------------------------------------------'''
    ''' 4. Calculate market shares						                        '''
    '''-------------------------------------------------------------------------'''

    # Calculate individual's utility
    u_choice = np.zeros((sum(list_J), N)) + np.repeat(delta, N, axis=1)
    for i in range(K_rc):
        u_choice += x_rc[:, i: i + 1] * v[:, i * N: (i + 1) * N]

    # Get CCPs
    ccp = get_ccp(u_choice, T, list_J, N)

    # Calculate market shares
    sigma = np.hstack([ccp[sum(list_J[:t]): sum(list_J[:t + 1]), :].mean(1) for t in range(T)])
    sigma_0 = np.hstack([1 - sum(sigma[sum(list_J[:t]): sum(list_J[:t + 1])]) for t in range(T)])
    sigma_0 = np.hstack([sigma_0[t: t + 1].repeat(list_J[t]) for t in range(T)])

    # if price_rc=True, put a random coefficient on price
    if price_rc:
        beta_p = beta_true[0] + v[:, 0]
    else:
        beta_p = beta_true[0]

    return {'market_ids': ids_market, 'price': price, 'x_1': x_1, 'x_2': x_2,
            'w': w, 's': sigma[:, np.newaxis], 's_0': sigma_0[:, np.newaxis], 'ccp': ccp, 'price_coef': beta_p, 'xi': xi}


def get_mat_diff_t(x):
    """
    create characteristic differences for one market
    """
    (J, K) = x.shape
    """mat_diff_t = np.empty((0, K))
    for j in range(J):
        #mat_diff_t[current : current + J, :] = x[j, :] - np.delete(x, j, axis=0)
        mat_diff_t = np.append(mat_diff_t, x - x[j, :], axis = 0)"""
    mat_diff_t = np.tile(x.T, J).T - np.repeat(x, J, axis=0)
    return mat_diff_t


def get_diff(ids_market, x, s=None):
    K = x.shape[1]
    mat_diff = []
    for t in range(min(ids_market), max(ids_market) + 1):
        J_t = int(sum(ids_market == t))
        x_t = x[ids_market == t, :]
        mat_diff_t = get_mat_diff_t(x_t)
        mat_diff_t = mat_diff_t.reshape(J_t, J_t * K)
        if s is not None:
            s_t = s[ids_market == t, :]
            s_t_mat = np.repeat(s_t.T, J_t, axis=0)
            mat_diff_t = np.c_[s_t_mat, mat_diff_t]
        mat_diff.append(mat_diff_t)
    mat_diff = np.concatenate(mat_diff, axis=0)
    return mat_diff


def data_transform(data):

    ### 1. stransform data ###
    # extract the data
    x_1 = data['x_1']
    x_2 = data['x_2']
    w = data['w']
    s = data['s']
    s_0 = data['s_0']
    p = data['price']
    market_ids = data['market_ids']
    xi = data['xi']

    # construct data for estimation
    # assuming price enters with RC
    Y = (np.log(s / s_0) - x_1).flatten()
    g = Y - xi.flatten()
    X = get_diff(market_ids, np.c_[p, x_2], s)
    print('Dimension of X vector: %s' % X.shape[1])
    Z = get_diff(market_ids, np.c_[x_1, x_2, w])
    print('Dimension of Z vector: %s' % Z.shape[1])

    ### 2. split data into train, dev, and test subsamples ###

    J = int(sum(market_ids == 0))  # all markets of the same length
    T = int(len(market_ids) / J)
    # 50% train, 25% dev, 25% test
    T_train = T // 2
    T_dev = (T - T_train) // 2

    # indices for splits
    train_id = range(T_train * J)
    dev_id = range(T_train * J, (T_train + T_dev) * J)
    test_id = range((T_train + T_dev) * J, T * J)

    # do splits
    X_train, Z_train, Y_train, g_train = X[train_id, :], Z[train_id, :], Y[train_id], g[train_id]
    X_dev, Z_dev, Y_dev, g_dev = X[dev_id, :], Z[dev_id, :], Y[dev_id], g[dev_id]
    X_test, Z_test, Y_test, g_test = X[test_id, :], Z[test_id, :], Y[test_id], g[test_id]

    # store as dictionaries
    data_train = {'X': X_train, 'Y': Y_train, 'Z': Z_train, 'g': g_train}
    data_dev = {'X': X_dev, 'Y': Y_dev, 'Z': Z_dev, 'g': g_dev}
    data_test = {'X': X_test, 'Y': Y_test, 'Z': Z_test, 'g': g_test}

    return {'train': data_train, 'dev': data_dev, 'test': data_test}


