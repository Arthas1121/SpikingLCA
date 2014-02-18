# -*- coding: utf-8 -*-

import numpy as np
import os

HOME = os.environ["HOME"]

# -----------------------------------------------------------------------------
def load_pickle(PATH):
# -----------------------------------------------------------------------------
    """ Loads the pickled data. """
    import cPickle as pickle
    import tempfile
    
    if not ('@' in PATH and ":" in PATH):
        return pickle.load(open(PATH))
    else:
        file_name=PATH.split(":")[1].rsplit("/")[-1]
        tmp_path=tempfile.mkdtemp()
        os.system("scp {} {}".format(PATH, tmp_path))
        result = pickle.load(open(tmp_path+'/'+file_name))
        rm_rf(tmp_path)
        return result

def choose_unique(k, the_list):
    """
    Chose k unique elements of a list the_list. It preserves the order.
    """
    n = len(the_list)
    assert k <= n
    chosen_idx = []
    while len(chosen_idx) < k:
        d = len(chosen_idx)
        new_candidates = list(np.random.random_integers(0, n-1, (k-d)))
        chosen_idx = list(set(chosen_idx + new_candidates))
    chosen_idx.sort()
    return [the_list[i] for i in chosen_idx]

def generative_model_data(Ndic, Ndim, k, positive_phi=False, positive_input=False, identity_dic=False, seed=None):
    
    if seed:
        np.random.seed(seed=seed)

    if not positive_phi:
        b = np.random.randn(Ndic, Ndim)
    else:
        b = np.abs(np.random.randn(Ndic, Ndim))

    ### Special case - identity dictionary
    if identity_dic:
        assert Ndic==Ndim, "you probably want identity_dic set to False."
        b = np.identity(Ndic)	
    ### end of the special case!!!

    norms = np.apply_along_axis(np.linalg.norm, 1, b)
    b_n = b / norms.reshape(-1,1)
    phi = b_n.T
    
    # create a G matrix from the normalized dictionaries
    G = np.zeros((Ndic,Ndic))
    for i in xrange(Ndic):
        for j in xrange(i,Ndic):
            G[i][j] = np.dot(b_n[i], b_n[j])
    G = G + G.T - 2*np.identity(Ndic)
    G[np.abs(G)<0.00000000001] = 0.
    
    # create the sparse vector (ground truth) and use it to generate input vector i_stim
    dict_idx = choose_unique(k, np.arange(Ndic))
    sparse_vector = np.zeros(Ndic)
    for i in dict_idx:
        if positive_input:
            sparse_vector[i] = np.abs(np.random.randn())
        else:
            sparse_vector[i] = np.random.randn()
    
    #sparse_vector_norm = 1.*x/np.linalg.norm(sparse_vector)
    x = np.dot(phi, sparse_vector)
    i_stim = np.dot(phi.T, x)

    result = {"G": G,
              "phi": phi,
              "a": sparse_vector,
              "x": x,
              "i_stim": i_stim,
              }

    return result

def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm: 
    	return 1.*vec/norm
    return vec

def super_normalize(vec):
    """
    Make the vector 0 mean with var=1. 
    """
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm: 
        vec = vec - np.mean(vec)
        vec = 1.*vec/np.std(vec)
    return vec

def rgb_to_grey(vec):
    """
    Turns the vector containing an rgb image (1/3 of vector red, 
    1/3 green, 1/3 blue) into a grey vector of length 1/3 of vec.
    """

    assert not len(vec) % 3
    res_len = len(vec)/3
    vec_r = vec.reshape((3,res_len))
    return np.dot(vec_r.T, [ 0.2989, 0.5870, 0.1140])
   
def plottable_rgb_matrix(vec, dim):
    assert not len(vec) % 3
    res_len = len(vec)/3
    assert res_len == (dim[0]*dim[1])
    vec_t = vec.reshape((3, res_len)).T
    vec_t = vec_t.reshape((1, len(vec)))[0]
    return vec_t.reshape((dim[0], dim[1] ,3))
