import numpy as np
import matplotlib.pyplot as pyl
import matplotlib.cm as cm
import LCA.LCA_common as cmn
from LCA.LCA_analysis import dict_set_at_idx

def plot_edge(SMs, i):
    figure()
    pyl.plot(-60-(1000.*SMs[0][i]), label="Vrest-vm")
    pyl.plot(SMs[1][i], label="ge")
    pyl.plot(SMs[2][i], label="gi")
    pyl.plot(SMs[4][i], label="i_s")
    pyl.legend()

def plot_a_evolutions(SMs, N):
    [pyl.plot(SMs[3][i][::10], "blue") for i in xrange(N)]

def plot_reconstruction_error(input_factors, nr_active, error, Ndic):
    pyl.figure(figsize=(12,4))
    pyl.subplot(1,2,1)
    pyl.plot(input_factors, error, label=r"$\|x-(\phi a)_{op}\|$")
#    pyl.xlabel(r"input factor")
    pyl.plot(input_factors, 1.-1.*np.array(nr_active)/Ndic, label=r"sparsness")
    leg = pyl.legend(loc="best")
    leg.draw_frame(False)
    pyl.subplot(1,2,2)
    pyl.plot(1.-1.*np.array(nr_active)/Ndic, error, label=r"$\|x-(\phi a)_{op}\|$", marker="o")
    pyl.xlabel("sparsness")
    leg = pyl.legend(loc="best")
    leg.draw_frame(False)

def plot_rate_spiking_comparison(rres, sres):
    pyl.plot(1.-1.*np.array(rres["nr_act"])/rres["pars"]["Ndic"], rres["err"], label="rate")
    pyl.plot(1.-1.*np.array(sres["nr_act"])/sres["pars"]["Ndic"], sres["err"], label="spiking")
    pyl.plot([1.-1.*rres["pars"]["k"]/rres["pars"]["Ndic"]]*2, [0,1], linestyle="--", color="gray", label="completness border")
    pyl.xlabel(r"Sparsness $(1-N_{active}/N_{dic})$")
    pyl.ylabel(r"$\|x-(\phi a)_{op}\|$")

def visualize_vector_comparison(vec, vec_comp=[], normalize=True, position="horizontal", scale=1.):
    assert len(np.shape(vec)) == 2, "wrong vector dimensions."
    
    vec_plt = vec.copy()

    if normalize:
        vec_plt = np.array([cmn.normalize(v) for v in vec_plt])
        vec_comp = cmn.normalize(vec_comp)
        vmin = np.min([np.min(vec_plt[-1]), np.min(vec_comp)])
        vmax = np.max([np.max(vec_plt[-1]), np.max(vec_comp)])
        
    dims = np.shape(vec_plt)
    if not len(vec_comp) == 0:
        if position == "horizontal":
            fig_size = (scale*dims[1]/2, scale*dims[0]/6)
            plt_pos1 = (1,2,1)
            plt_pos2= (1,2,2)
        else:
            fig_size = (scale*dims[1]/3, scale*dims[0]/2)
            plt_pos1 = (2,1,1)
            plt_pos2 = (2,1,2)
    else:
        fig_size = (scale*dims[1]/3, scale*dims[0]/3)
   
    pyl.figure(figsize=fig_size)
    if not len(vec_comp) == 0:
        pyl.subplot(*plt_pos1)
    if normalize:
        pyl.imshow(vec_plt, interpolation="nearest", cmap = cm.Greys_r, vmin=vmin, vmax=vmax)
    else:
        pyl.imshow(vec_plt, interpolation="nearest", cmap = cm.Greys_r)
    if not len(vec_comp) == 0:
        pyl.subplot(*plt_pos2)
        if normalize:
            pyl.imshow([vec_comp for i in xrange(len(vec))], interpolation="nearest", cmap = cm.Greys_r, vmin=vmin, vmax=vmax)
        else:
            pyl.imshow([vec_comp for i in xrange(len(vec))], interpolation="nearest", cmap = cm.Greys_r)

#######################################################
######### PetaVision related plotting #################
#######################################################


def plot_dict_set(d, x, y, cmap=cm.Greys, interpolation="nearest"):
    """
    Plots the set of dictionary elements. d is the set of dictionary elements 
    in PetaVision coordinate system (x - columns, y - rows).
    """
    assert x*y > len(d)
    for i in xrange(len(d)):
        pyl.subplot(x,y,i+1)
        pyl.imshow(d[i].T, cmap=cmap, interpolation=interpolation)
        pyl.axis('off')
        
def plot_dict_evolution(dict_data, idxs=[], shape=None, outputdir="./", base_file_name="images", image_format="png", create_movie=True):
    """
    Creates and saves the visualizations of the dictionaries. Optionally creates a movie as well.
    
    dict_data: The dictionaries as loaded from the .pvp file into python.
    idxs: A list of indexes of those dictionaries to be plotted. If empty plot them all.
    shape: The shape of the image - a tuple of 2 elements.
    NOTE: Currently only supports reconstruction of one postsynaptic feature!

    NOTE: more on the movie creation: https://trac.ffmpeg.org/wiki/x264EncodingGuide
    NOTE: more on displaying the movie in iPython Notebook: 
    http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb
    """
    
    assert np.max(idxs) <= len(dict_data)
    
    if create_movie:
        assert image_format=="png", "Image format must be png if you want to create the movie."
    
    if not shape:
        nr_of_dicts = len(dict_set_at_idx(dict_data, idxs[0]))
        shape = (np.ceil(np.sqrt(nr_of_dicts)),)*2
        
    assert shape[0]*shape[1] >= len(dict_set_at_idx(dict_data, idxs[0]))
    
    ### Plotting the images ###
    for j, idx in enumerate(idxs):
        the_dict = dict_set_at_idx(dict_data, idx)
        fig = pyl.figure(figsize=(shape[0]*10/6,)*2)
        plot_dict_set(the_dict, shape[0], shape[1])
        pyl.savefig("{0}/{1}-{2}.{3}".format(outputdir, base_file_name, j+1, image_format))
        pyl.close(fig)
    
    ### Movie from images ###
    if create_movie:
        from subprocess import call
        
        call(["ffmpeg", "-r", "2",\
        "-i", "{folder}/{fn}-%d.png".format(fn=base_file_name, folder=outputdir),\
        "-c:v", "libx264", "-r", "30",  "-pix_fmt", "yuv420p", "-s", "480x480",\
        "{folder}/movie.mp4".format(folder=outputdir)])
