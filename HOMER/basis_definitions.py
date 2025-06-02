import jax.numpy as jnp
import jax
import numpy as np

#we can refactor these function
def N2_weights(w0, w1):
    BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
             [2, 0], [3, 0], [2, 1], [3, 1],
             [0, 2], [1, 2], [0, 3], [1, 3],
             [2, 2], [3, 2], [2, 3], [3, 3]]
    w_list = []
    for ii in BPInd:
        w_list.append(w0[:, ii[0]] * w1[:, ii[1]])
    weights = jnp.column_stack(w_list)
    return weights

def N3_weights(w0, w1, w2):
    BPInd = [
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        [2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0],
        [2, 0, 1], [3, 0, 1], [2, 1, 1], [3, 1, 1],
        [0, 2, 0], [1, 2, 0], [0, 3, 0], [1, 3, 0],
        [0, 2, 1], [1, 2, 1], [0, 3, 1], [1, 3, 1],
        [2, 2, 0], [3, 2, 0], [2, 3, 0], [3, 3, 0],
        [2, 2, 1], [3, 2, 1], [2, 3, 1], [3, 3, 1],

        [0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2],
        [0, 0, 3], [1, 0, 3], [0, 1, 3], [1, 1, 3],
        [2, 0, 2], [3, 0, 2], [2, 1, 2], [3, 1, 2],
        [2, 0, 3], [3, 0, 3], [2, 1, 3], [3, 1, 3],
        [0, 2, 2], [1, 2, 2], [0, 3, 2], [1, 3, 2],
        [0, 2, 3], [1, 2, 3], [0, 3, 3], [1, 3, 3],
        [2, 2, 2], [3, 2, 2], [2, 3, 2], [3, 3, 2],
        [2, 2, 3], [3, 2, 3], [2, 3, 3], [3, 3, 3],
    ]
    w_list = []
    for ii in BPInd:
        w_list.append(w0[:, ii[0]] * w1[:, ii[1]] * w2[:, ii[2]])
    weights = jnp.column_stack(w_list)
    return weights



def H3H3_eval(eles, xis, param_list, ele_arrays):
    xis = xis.reshape(xis.shape[0]//2,2)
    ele_data = create_eledata(param_list, ele_arrays)
    prods = ele_data[eles, :, :] #shape ((xis, 3, 16))
    phi_weights = make_std_weights(xis) # shape (xis, 16)
    output = jnp.sum(prods * phi_weights[:, None, :], axis=-1).flatten()
    return output

def H3H3_single_ele(xis, params):
    """
    given the params associated with an element
    reshape the params to be of the form 3 x 16 
    """
    phi_weights = make_std_weights(xis)
    output = jnp.sum(params * phi_weights[:, None, :], axis=-1).flatten()
    return output


def make_std_weights(xis):
    W0 = H3(xis[:, 0])
    W1 = H3(xis[:, 1])
    BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
             [2, 0], [3, 0], [2, 1], [3, 1],
             [0, 2], [1, 2], [0, 3], [1, 3],
             [2, 2], [3, 2], [2, 3], [3, 3]]
    w_list = []
    for ii in BPInd:
        w_list.append(W0[:, ii[0]] * W1[:, ii[1]])
    weights = jnp.column_stack(w_list)
    return weights

def H3H3_sobolov_single_ele(xis, params):
    sobolov_arrays = make_drv_weights(xis)  # shape (xis, 16)
    derivs = jnp.sum(params * sobolov_arrays[:, :, None, :], axis=-1)
    individual_score = jnp.sum(jnp.abs(derivs), axis=-1)
    final_score = jnp.sum(individual_score, axis=0)
    return final_score

def H3H3_sobolov_prior_single_ele(xis, params, prior_derivs):
    """
    The code evaluates the sobolov with reference to reference curvatures
    :param xis:
    :param params:
    :param original:
    :return:
    """
    sobolov_arrays = make_drv_weights(xis)  # shape (xis, 16)
    derivs = jnp.sum(params * sobolov_arrays[:, :, None, :], axis=-1)
    # prior_derivs = jnp.sum(prior_params * sobolov_arrays[:, :, None, :], axis=-1)
    individual_score = jnp.sum(jnp.abs(derivs - prior_derivs), axis=-1)
    final_score = jnp.sum(individual_score, axis=0)
    return final_score

def H3H3_derivs_eval(eles, xis, param_list, ele_arrays):
    ele_data = create_eledata(param_list, ele_arrays)
    prods = ele_data[eles, :, :] #shape ((xis, 3, 16))
    sobolov_arrays = make_drv_weights(xis)  # shape (xis, 16)
    derivs = jnp.sum(prods[None] * sobolov_arrays[:, :, None, :], axis=-1)
    return derivs

def H3H3_sobolov_eval(eles, xis, param_list, ele_arrays, weights):
    ele_data = create_eledata(param_list, ele_arrays)
    prods = ele_data[eles, :, :] #shape ((xis, 3, 16))
    sobolov_arrays = make_drv_weights(xis)  # shape (xis, 16)
    derivs = jnp.sum(prods[None] * sobolov_arrays[:, :, None, :], axis=-1)
    individual_score = jnp.sum(jnp.abs(derivs), axis=-1) * weights[:, None]
    final_score = jnp.sum(individual_score, axis=0)
    # breakpoint()
    return final_score

def H3H3_sobolov_prior_eval(eles, xis, param_list, ele_arrays, weights, prior_derivs):
    ele_data = create_eledata(param_list, ele_arrays)
    prods = ele_data[eles, :, :] #shape ((xis, 3, 16))
    sobolov_arrays = make_drv_weights(xis)  # shape (xis, 16)
    derivs = jnp.sum(prods[None] * sobolov_arrays[:, :, None, :], axis=-1)
    # grab the priors through here
    individual_score = jnp.sum(jnp.abs(derivs - prior_derivs), axis=-1) * weights[:, None]
    final_score = jnp.sum(individual_score, axis=0)
    return final_score

def make_drv_weights(xis):
    W0_0 = H3(xis[:, 0])
    W1_0 = H3(xis[:, 1])
    W0_1 = H3d1(xis[:, 0])
    W1_1 = H3d1(xis[:, 1])
    W0_2 = H3d1d1(xis[:, 0])
    W1_2 = H3d1d1(xis[:, 1])

    BPInd = [[0, 0], [1, 0], [0, 1], [1, 1],
             [2, 0], [3, 0], [2, 1], [3, 1],
             [0, 2], [1, 2], [0, 3], [1, 3],
             [2, 2], [3, 2], [2, 3], [3, 3]]
    w10_l = []
    w01_l = []
    w20_l = []
    w02_l = []
    w11_l = []
 
    for ii in BPInd:
        w10_l.append(W0_1[:, ii[0]] * W1_0[:, ii[1]])
        w01_l.append(W0_0[:, ii[0]] * W1_1[:, ii[1]])
        w20_l.append(W0_2[:, ii[0]] * W1_0[:, ii[1]])
        w02_l.append(W0_0[:, ii[0]] * W1_2[:, ii[1]])
        w11_l.append(W0_1[:, ii[0]] * W1_1[:, ii[1]])

    w10 = jnp.column_stack(w10_l)
    w01 = jnp.column_stack(w01_l)
    w20 = jnp.column_stack(w20_l)
    w02 = jnp.column_stack(w02_l)
    w11 = jnp.column_stack(w11_l)
    return jnp.array((w10, w01, w20, w02, w11))

def H3(x:jnp.ndarray):
    #J is a 1D array of points to interpolate
    x2 = x*x
    Phi = jnp.column_stack([
        1-3*x2+2*x*x2,
        x*(x-1)*(x-1),
        x2*(3-2*x),
        x2*(x-1)
    ])
    return Phi

def H3d1(x: jnp.ndarray):
    """
    First derivative of the cubic-Hermite basis function.
    
    :param x: points to interpolate
    :type x: np array (npoints)
    :return: basis weights
    :rtype: np array(npoints, 4)
    """
    x2 = x*x
    Phi = jnp.column_stack([ \
        6*x*(x-1),
        3*x2-4*x+1,
        6*x*(1-x),
        x*(3*x-2)])
    return Phi

def H3d1d1(x):
    """
    First derivative of the cubic-Hermite basis function.
    
    :param x: points to interpolate
    :type x: np array (npoints)
    :return: basis weights
    :rtype: np array(npoints, 4)
    """
    Phi = jnp.column_stack([ \
        12*x-6,
        6*x-4,
        6-12*x,
        6*x-2]) 
    return Phi

if __name__ == "__main__":

    pi = np.pi
    Xn = np.array([
            [-pi, -pi, 0],
            [  0, -pi, 0],
            [ pi, -pi, 0],
            [-pi,   0, 0],
            [  0,   0, 0],
            [ pi,   0, 0],
            [-pi,  pi, 0],
            [  0,  pi, 0],
            [ pi,  pi, 0]
    ])

    Xn = np.random.random((9,3))

    deriv = np.random.random((3,3))
    m = morphic.Mesh()
    for i, xn in enumerate(Xn):
        xn_ch = np.append(xn[:, None], deriv, axis=1)
        m.add_stdnode(i+1, xn_ch, group='_default')

    m.add_element(0, ['H3', 'H3'], [1, 2, 4, 5])
    m.add_element(1, ['H3', 'H3'], [4, 5, 7, 8])
    m.add_element(2, ['H3', 'H3'], [2, 3, 5, 6])
    m.add_element(3, ['H3', 'H3'], [5, 6, 8, 9])
    m.generate()


    ele_ids = [0, 1, 2, 3]
    xis = np.random.random((10000, 2))
    res = m.evaluate(ele_ids, xis)

    #how do we flatten the data to the result.
    params = np.array(m.core.P)
    ele_array = np.array(m.core.EMap)

    # ele_data = create_eledata(params, ele_array)
    h3h3_eles = np.repeat(ele_ids, xis.shape[0]) 
    h3h3_xis = np.tile(xis, (len(ele_ids),1)).flatten()

    # res_2 = H3H3_eval(h3h3_eles, h3h3_xis, params, ele_array)
    jax_h3h3 = jax.jit(H3H3_eval)
    # jax_sob_h3h3 = jax.jit(H3H3_sobolov_eval)

    res_2 = jax_h3h3(h3h3_eles, h3h3_xis, params, ele_array).reshape((h3h3_eles.shape[0], 3))
    weights = np.array([0.1, 0.1, 0.01, 0.01, 0.01])
    # sob = jax_sob_h3h3(h3h3_eles, h3h3_xis, params, ele_array, weights)

    difs = res - res_2
    print(f" found a maximum difference between methods of {np.max(difs)}")
    # print(difs)
    # breakpoint()

    print("Morphic")
    benchmark(lambda: m.evaluate(ele_ids, xis), repeats=10, mode='ms')
    # print("Jax raw")
    # # benchmark(lambda: H3H3_eval(h3h3_eles, h3h3_xis, params, ele_array))
    print("Jax compiled")
    benchmark(lambda: jax_h3h3(h3h3_eles, h3h3_xis, params, ele_array), repeats=100, mode='ms')

    # jax_test_grad = get_h3h3_grad_eval(h3h3_xis)

    # shape = jax_test_grad(h3h3_eles, h3h3_xis, params, ele_array)
    # print(shape.shape)

    # print("Jax compiled CUDA gradient")
    # benchmark(lambda: jax_test_grad(h3h3_eles, h3h3_xis, params, ele_array), repeats=100, mode='ms')

    # print("Jax compiled CUDA sobolov")
    # benchmark(lambda: jax_sob_h3h3(h3h3_eles, h3h3_xis, params, ele_array, weights), repeats=100, mode='ms')










