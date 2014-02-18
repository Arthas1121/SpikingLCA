import numpy as np
from neurovivo.common.spike_train import SpikeTrain

def reconstruction(b_n, M, t_end):
    """
    Returns a reconstructed input vector by using the freq of spiking
    in the last second.
    """
    a = number_of_spikes_last_second(M, t_end)
    rec = np.dot(b_n.T,a)
    return rec

def reconstruction_error(b_n, x, rec, fact=1.):
    assert np.linalg.norm(x)==1.
    if not np.linalg.norm(rec) == 0:
        rec_n = rec/np.linalg.norm(rec)
    else: 
        rec_n = rec
    rec_op = fact*np.dot(rec_n, x) * rec_n
    return np.linalg.norm(x - rec_op)
    
def number_of_active_neurons(M, min_time):
    """
    Returns the number of neurons that spiked since min_time until 
    the end of the simulation.
    """
    n_active = 0
    for st in xrange(len(M.source)):
        if len(M[st][M[st]>min_time/1000.])>0:
            n_active += 1
    return n_active

def number_of_spikes_last_second(M, t_end):
    """
    Returns list of number of spikes per neuron in the last second 
    before t_end.
    """
    result = []
    for st in xrange(len(M.source)):
        result.append(len(M[st][M[st]>(t_end/1000.-1.)]))
    return np.array(result).T

def get_dynamic_a(M, window, dt, t_end):
    """
    Gives the dynamic a value for all the neurons.  
    """
    aas = []
    for st in xrange(len(M.source)):
        spike_train = SpikeTrain(1000*M[st],total_time=t_end)
        aas.append(spike_train.dynamic_calculations(dt=dt,window=window, calculation="rate"))
    return np.array(aas)

def opt_error(vec1, vec2):
    if not np.linalg.norm(vec1) == 0:
        vec1 = vec1/np.linalg.norm(vec1)
    if not np.linalg.norm(vec2) == 0:
        vec2 = vec2/np.linalg.norm(vec2)
    vec2_op = np.dot(vec1,vec2)*vec2
    return np.linalg.norm(vec1-vec2_op)

def error_of_normalized(vec1, vec2):
    if not np.linalg.norm(vec1) == 0:
        vec1 = vec1/np.linalg.norm(vec1)
    if not np.linalg.norm(vec2) == 0:
        vec2 = vec2/np.linalg.norm(vec2)
    return np.linalg.norm(vec1-vec2)

def get_dynamic_error(dynamic_a, GM):
    """
    Returns the dynamic reconstruction error. 
    """
    dynamic_rec = np.dot(GM["phi"], dynamic_a)
    return [opt_error(GM["x"], dr) for dr in dynamic_rec.T] 

def reconstruction_evaluation(a_LCA, GM, pars):

    x_n = GM["x"]/np.linalg.norm(GM["x"])
    rec = np.dot(GM["phi"], a_LCA)
    rec_n = np.zeros(len(rec))
    if not np.linalg.norm(rec) == 0:
        rec_n = rec/np.linalg.norm(rec)
    rec_opt = np.dot(rec_n, x_n) * rec_n

    opt_error = np.linalg.norm(rec_opt - x_n)
    
    result={"normalized error": 1.*np.linalg.norm(GM["x"] - rec)/np.linalg.norm(GM["x"]),
            "normalized a diff": 1.*np.linalg.norm(a_LCA-GM["a"])/np.linalg.norm(GM["a"]),
            "nonzero": np.count_nonzero(a_LCA),
            "error + thr a1": np.linalg.norm(GM["x"] - rec) + pars["threshold"] * np.sum(np.abs(a_LCA)),
            "thr a_t1": pars["threshold"] * np.sum(np.abs(GM["a"])),
            "opt_error": opt_error,
            }
    return result

