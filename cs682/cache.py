def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    num = int(C / G)
    x = np.reshape(x, (num * H * W, N * G))
    mean = np.mean(x, axis=0)
    x_mean = x - mean
    var = np.var(x, axis=0)
    sqrtvar = np.sqrt(var + eps)
    normalized_data = x_mean / sqrtvar
    normalized_data = np.reshape(normalized_data.T, (N, C, H, W))
    out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * normalized_data + beta[np.newaxis, :, np.newaxis, np.newaxis]
    cache = {
        'x_mean': x_mean,
        'normalized_data': normalized_data,
        'gamma': gamma,
        'sqrtvar': sqrtvar,
        'var': var,
        'eps': eps,
        'G': G
    }
    # cache = (normalized_data, gamma, x_mean, var_invert, sqrtvar, var, eps, G)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N, C, H, W = dout.shape
    gamma = cache.get('gamma')
    x_mean = cache.get('x_mean')
    sqrtvar = cache.get('sqrtvar')
    G = cache.get('G')
    var = cache.get('var')
    eps = cache.get('eps')
    normalized_data = cache.get('normalized_data')
    dxhat = dout * gamma[np.newaxis, :, np.newaxis, np.newaxis]

    # Set keepdims=True to make dbeta and dgamma's shape be (1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * normalized_data, axis=(0, 2, 3), keepdims=True)

    # Reshape and transpose back
    dxhat = np.reshape(dxhat, (N * G, C // G * H * W)).T
    xhat = np.reshape(normalized_data, (N * G, C // G * H * W)).T

    Nprime, Dprime = dxhat.shape

    dx = 1.0 / Nprime / sqrtvar * (Nprime * dxhat - np.sum(dxhat, axis=0) - xhat * np.sum(dxhat * xhat, axis=0))

    dx = np.reshape(dx.T, (N, C, H, W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
