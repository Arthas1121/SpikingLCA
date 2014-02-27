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

##########################################
##### PetaVision related analysis ########
##########################################

def dict_set_at_idx(data, idx):
    """
    data: The dictionaries as loaded from the .pvp file into python from the
          <kernel-name>.pvp .
    idx: The timestamp index to be extracted.
    
    Returns the np.array of dictionaries in the shape (nxp, nyp, nfp, nf).
    """
    return data[idx][0][1][0][0]


def sparse_to_full(sparse_data, header, idx=-1):
    """
    Takes the sparse data and header which are read from the sparsly written
    full activity .pvp file and returns the full blown activity matrix in 
    the form (nx, ny, nf) for the time index idx (defalut is the activity at
    the last recorded time). 
    
    The indexes in the sparse_data are increasing in the PetaVision order 
    (nf, nx, ny - in the order of fastest changing to the slowets changing).
    
    It returns the 3D array in the form (nx, ny, nf), where the dimension 
    values are specified in the header.
    """
    
    shape = (header["nx"], header["ny"], header["nf"])
    data = sparse_data[idx][0][1]

    ### replace the non-existing values with zeroes
    vector = np.zeros(shape[0]*shape[1]*shape[2])
    for i in xrange(len(data)):
        vector[int(data[i][0])] = data[i][1]

    ### return the reshaped 3D vector which dimensions are (nx, ny, nf)
    return np.rollaxis(vector.reshape(shape[1], shape[0], shape[2]), 1)


def reconstruct_image(activities, dictionary, nx_rel_scale=1., ny_rel_scale=1.):
    """
    Given the activities of the presynaptic layer and a dictionary it 
    reconstructs the postsynaptic activity in the form (x, y, nfp).

    i.e. if the presynaptic layer is V1 and the dictionary are the weigths
    V1->LGN it reconstructs the image projected to LGN -> thus the function
    name.
    """
    
    assert len(activities[0,0,:]) == len(dictionary[0,0,0,:])
    
    nxp, nyp, nfp, nf = dictionary.shape
    nx, ny, nf = activities.shape
    
    # Calculate the extended postsynaptic surface based on the presynaptic 
    # scales, dimensions and postsynaptic patch size.
    base_x = int(nxp + 1.*(nx-1)/nx_rel_scale)
    base_y = int(nyp + 1.*(ny-1)/ny_rel_scale)
    
    reconstruction = np.zeros((base_x, base_y, nfp))
    ### Loop over all features and their corresponding nodes.
    for f in xrange(nf):
        for fp in xrange(nfp):
            dic = dictionary[:, :, fp, f]
            act = activities[:,:,f]
            for i in xrange(nx):
                for j in xrange(ny):
                    if act[i,j]:
                        xmin = int(i/ny_rel_scale)
                        xmax = xmin + nxp
                        ymin = int(j/nx_rel_scale)
                        ymax = ymin + nyp
                        reconstruction[xmin:xmax, ymin:ymax, fp] += act[i,j]*dic

    # Crop and return the image
    padx = int(nxp/2. - 1./(2*nx_rel_scale))
    pady = int(nyp/2. - 1./(2*ny_rel_scale))

    return reconstruction[pady:-pady, padx:-padx]

def dict_from_flat_data(flat_data, header):
    """
    Takes the flat_data and header which are results of readpvpfile on the dict
    files in the checkpoints (the ones ending with _W). For some reason the
    oct2py flattens the dictionary, so we have to reshape it.
    """

    desired_shape = (header["nxp"], header["nxp"], header["nfp"], header["nf"])
    return np.array(flat_data[0]["values"]).reshape(desired_shape)


def list_position_to_3D(idx, shape):
    """
    Returns the 3D index for a location based on the list position from PVP file.
    idx: integer
    shape: (nx, ny, nf)
    
    Returns the position in 3D array corresponding to the idx.

    NOTE: The function does not check for validity of the inputs.
    """
    ny = idx/(shape[0]*shape[2])
    nx = (idx%(shape[0]*shape[2]))/shape[2]
    nf = (idx%(shape[0]*shape[2]))%shape[2]
    
    return (nx, ny, nf)


def spike_population_from_pvp(data, header, dt):
    """
    Takes the sparse data from the spiking layer and returns SpikePopulation object.
    
    data:   Data read out from the .pvp activity file of a spiking object. The format 
            must be sparse and if you do not write for every step you are probably missing
            spikes.
    header: The header of that same .pvp file.
    dt:     
    """
    from neurovivo.common.spike_population import SpikePopulation
    from neurovivo.common.spike_train import SpikeTrain

    nx, ny, nf = int(header["nx"]), int(header["ny"]), int(header["nf"])
    shape = (nx, ny, nf)
    
    spike_times = [[] for _ in xrange(np.product(shape))]
    for t_idx in xrange(len(data)):
        # if not all silent at this time fill in the spike_times
        if np.any(data[t_idx][0][1]):
            [spike_times[int(n_idx)].append(t_idx*dt) for n_idx in data[t_idx][0][1].T[0]]
    
    return SpikePopulation([SpikeTrain(st) for st in spike_times])
