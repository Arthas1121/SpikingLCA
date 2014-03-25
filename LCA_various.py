# -*- coding: utf-8 -*-
"""
A set of very specific functions and parameters for the spiking LCA project.
"""

import numpy as np
from subprocess import call, STDOUT
from shutil import move
import LCA.LCA_common as cmn
from LCA.LCA_common import create_params_file
import os

default_CIFAR10Spiking_v4_parameters = {
                      "comment": "Default parameter set.",
                      "output_path": "\"/home/mpelko/data/CIFAR10Spiking/v4/\"",
                      "stop_time": 4000.0,
                      "display_period": 500.0,
                      "spike-count_dt": 500.0,
                      "retina_foreground": 1000,
                      "image_list_path": "\"/home/mpelko/ws/CIFAR10Spiking/input/image_filelist.txt\"",
                      "V12res_weights_file_pos": "\"/home/mpelko/ws/CIFAR10Spiking/input/weights/V1_to_residual_W.pvp_pos\"",
                      "V12res_weights_file_neg": "\"/home/mpelko/ws/CIFAR10Spiking/input/weights/V1_to_residual_W.pvp_neg\"",
                      "res2V1_weights_file_pos": "\"/home/mpelko/ws/CIFAR10Spiking/input/weights/residual_to_V1_W.pvp_pos\"",
                      "res2V1_weights_file_neg": "\"/home/mpelko/ws/CIFAR10Spiking/input/weights/residual_to_V1_W.pvp_neg\"",
                      "V12res_on_inh_W": 10., 
                      "V12res_on_exc_W": 10., 
                      "V12res_off_inh_W": 10., 
                      "V12res_off_exc_W": 10., 
                      "res_on2V1_exc_W": 10., 
                      "res_on2V1_inh_W": 10., 
                      "res_off2V1_inh_W": 10., 
                      "res_off2V1_exc_W": 10., 
                      "V12V1_W": 10.,
                                     }

default_CIFAR10Spiking_v3_parameters = {
                      "comment": "Default parameter set.",
                      "output_path": "\"/home/mpelko/data/CIFAR10Spiking/v3/\"",
                      "stop_time": 4000.0,
                      "display_period": 500.0,
                      "spike-count_dt": 50.0,
                      "image_list_path": "\"/home/mpelko/ws/CIFAR10Spiking/input/image_filelist.txt\"",
                      "V12res_weights_file_pos_exc": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_pos_10\"",
                      "V12res_weights_file_pos_inh": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_pos_10\"",
                      "V12res_weights_file_neg_exc": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_neg_10\"",
                      "V12res_weights_file_neg_inh": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_neg_10\"",
                      "V12V1_W": .5,
                                     }

default_CIFAR10Spiking_parameters = {
                      "comment": "Default parameter set.",
                      "output_path": "\"/home/mpelko/data/CIFAR10Spiking/\"",
                      "stop_time": 1000.0,
                      "display_period": 500.0,
                      "image_list_path": "\"/home/mpelko/ws/CIFAR10Spiking/input/image_filelist.txt\"",
                      "init_weights_file_pos_exc": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_pos_10\"",
                      "init_weights_file_pos_inh": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_pos_20\"",
                      "init_weights_file_neg_exc": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_neg_10\"",
                      "init_weights_file_neg_inh": "\"/home/mpelko/data/CIFAR10Rate/checkpoints/Checkpoint2000000/V1_to_residual_W.pvp_neg_20\"",
                                     }

default_MeasuringFICurves_parameters = {
                      "comment": "Default parameter set.",
                      "output_path": "\"/home/mpelko/data/SimpleSpikingNetwork/MeasuringFICurves/\"",
                      "stop_time": 5000.0,
                      "inp2ret_W": 1.0,
                      "ret2LIF_W": 1.0,
                      "ret2LIF_channel": 0,
                      "ret2LIF2_W": 0.,
                      "ret2LIF2_channel": 1,
                      "deltaGIB": 1.
                                     }

def run_CIFAR10Spiking(params, version=1):
    """
    Runs the CIFAR10Spiking simulation with the given parameters in params.
    Returns the path to the output of the simulation.
    """    
    
    template_path = "/home/mpelko/ws/CIFAR10Spiking/input/CIFAR10Spiking_template_v{}.params".format(version)

   
    import random

    tmp_params_name = "{:x}.params".format(random.randrange(16**30))
    tmp_params_path = create_params_file(template_path, params, "/home/mpelko/tmp/{}".format(tmp_params_name))
    tmp_file_name = "run_{:x}.log".format(random.randrange(16**30))
    tmp_file_path = cmn.HOME+"/tmp/" + tmp_file_name
    
    with open(tmp_file_path, "w") as logfile:
        res = call(["/home/mpelko/ws/CIFAR10Spiking/Debug/CIFAR10Spiking", "-p",\
        tmp_params_path], stdout=logfile, stderr=STDOUT)
    
    if not res == 0:
        print "The simulation ran failed. I don't know how to get the error message, so you have to run in manually."
   
    try:
        os.remove(params["output_path"][1:-1] + "/run.log")
    except:
       pass
    
    try:
        os.remove(params["output_path"][1:-1] + "/params.params")
    except:
        pass

    move(tmp_params_path, params["output_path"][1:-1] + "/params.params")
    move(tmp_file_path, params["output_path"][1:-1] + "/run.log")
    
    return params["output_path"][1:-1]

def run_MeasuringFIcurves(params):
    """
    Runs the MeasuringFIcurves simulation with the given parameters in params.
    Returns the path to the output of the simulation.
    """    

    template_path = "/home/mpelko/ws/SimpleSpikingNetwork/input/MeasuringFIcurves_template.params"
    params_path = create_params_file(template_path, params, "/home/mpelko/tmp/params.params")
    
    call(["/home/mpelko/ws/SimpleSpikingNetwork/Debug/SimpleSpikingNetwork", "-p", params_path])
    
    move(params_path, params["output_path"][1:-1] + "/params.params")
    
    return params["output_path"][1:-1]
