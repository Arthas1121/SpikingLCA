import numpy as np
import brian as br
from neurovivo.common import Bunch
from LCA_common import generative_model_data

def simulate_spiking3(parameters):
    
    x = Bunch(parameters)
    tau_m_inv = 1./(x.tau_m*br.ms)
    tau_e_inv = 1./(x.tau_e*br.ms)
    tau_i_inv = 1./(x.tau_i*br.ms)
    tau_a_inv = 1./(x.tau_a*br.ms)

    GM = generative_model_data(x.Ndic, x.Ndim, x.k, positive_phi=x.positive_phi, positive_input=x.positive_input, seed=x.seed)
    br.clear(all=True) # clears away all the brian objects from previous simulations.
        
    eqs = '''
          dvm/dt=((x.v_rest*br.mV-vm) + ge*(x.v_rev_e*br.mV-vm) + gi*(x.v_rev_i*br.mV-vm) + i_s)*tau_m_inv :br.mV
          dge/dt = -ge*tau_e_inv :1
          dgi/dt = -gi*tau_i_inv :1
          da/dt = -a*tau_a_inv   :br.Hz
          i_s                 :br.mV
          '''
        
    simulation_clock = br.Clock(dt=x.dt*br.ms)
    recording_clock = br.Clock(dt=1*br.ms)
    
    P = br.NeuronGroup(x.Ndic, model=eqs, threshold=x.v_thr*br.mV, reset=x.v_reset*br.mV, refractory=x.t_refr*br.ms, clock=simulation_clock)
    P.vm =  (((np.random.random((x.Ndic))-0.5) * 0.5 * (x.v_rest-x.v_thr)) + x.v_rest) * br.mV
    P.i_s = x.input_factor * GM["i_stim"] * br.mV
    P.a = 0

    G_exc = GM["G"].copy()
    G_exc[G_exc>0]=0
    G_exc = -1.*G_exc
    G_inh = GM["G"].copy()
    G_inh[G_inh<0]=0
    
    Ce = br.Connection(P, P, 'ge')
    Ce.connect(P, P, x.w_e*G_exc)
    Ci = br.Connection(P, P, 'gi')
    Ci.connect(P, P, x.w_i*G_inh)
    
    Ca = br.IdentityConnection(P, P, 'a', weight=tau_a_inv)
            
    M = br.SpikeMonitor(P)
    
    SMs = []
    SM_1 = br.StateMonitor(P, "vm", record=True, timestep=1, clock=recording_clock)
    SM_2 = br.StateMonitor(P, "ge", record=True, timestep=1, clock=recording_clock)
    SM_3 = br.StateMonitor(P, "gi", record=True, timestep=1, clock=recording_clock)
    SM_4 = br.StateMonitor(P, "a", record=True, timestep=1, clock=recording_clock)
    SM_5 = br.StateMonitor(P, "i_s", record=True, timestep=1, clock=recording_clock)
    SMs.append(SM_1)
    SMs.append(SM_2)
    SMs.append(SM_3)
    SMs.append(SM_4)
    SMs.append(SM_5)
    
    network = br.Network(P, Ce, Ci, Ca, M, *SMs)
        
    # starting the simulation
    network.run(x.t_end*br.ms)

    return SMs, M, GM

def simulate_spiking5(parameters):
    
    x = Bunch(parameters)
    tau_m_inv = 1./(x.tau_m*br.ms)
    tau_e_inv = 1./(x.tau_e*br.ms)
    tau_i_inv = 1./(x.tau_i*br.ms)
    tau_a_inv = 1./(x.tau_a*br.ms)
    
    assert x.GM, "generate the surrogate data and pass it into the fuction parameters."

    GM = x.GM 
    br.clear(all=True) # clears away all the brian objects from previous simulations.
        
    eqs = '''
          dvm/dt=((x.v_rest*br.mV-vm) + ge*(x.v_rev_e*br.mV-vm) + gi*(x.v_rev_i*br.mV-vm) + i_s)*tau_m_inv :br.mV
          dge/dt = -ge*tau_e_inv :1
          dgi/dt = -gi*tau_i_inv :1
          da/dt = -a*tau_a_inv   :br.Hz
          i_s                 :br.mV
          '''
        
    simulation_clock = br.Clock(dt=x.dt*br.ms)
    recording_clock = br.Clock(dt=1*br.ms)
    
    P = br.NeuronGroup(x.Ndic, model=eqs, threshold=x.v_thr*br.mV, reset=x.v_reset*br.mV, refractory=x.t_refr*br.ms, clock=simulation_clock)
    P.vm =  (((np.random.random((x.Ndic))-0.5) * 0.5 * (x.v_rest-x.v_thr)) + x.v_rest) * br.mV
    
    if x.tonic_input:
        P.i_s = x.input_factor_t * GM["i_stim"] * br.mV
    P.a = 0


    G_exc = GM["G"].copy()
    G_exc[G_exc>0]=0
    G_exc = -1.*G_exc
    G_inh = GM["G"].copy()
    G_inh[G_inh<0]=0
    
    Ce = br.Connection(P, P, 'ge')
    Ce.connect(P, P, x.w_e*G_exc)
    Ci = br.Connection(P, P, 'gi')
    Ci.connect(P, P, x.w_i*G_inh)
    
    Ca = br.IdentityConnection(P, P, 'a', weight=tau_a_inv)
    
    if not x.tonic_input:
	Ps = br.PoissonGroup(x.Ndic, rates = GM["i_stim"]*x.input_factor_s*br.Hz, clock=simulation_clock)
        Cs = br.IdentityConnection(Ps, P, 'ge', weight=x.w_s)    
        
    M = br.SpikeMonitor(P)
    
    SMs = []
    SM_1 = br.StateMonitor(P, "vm", record=True, timestep=1, clock=recording_clock)
    SM_2 = br.StateMonitor(P, "ge", record=True, timestep=1, clock=recording_clock)
    SM_3 = br.StateMonitor(P, "gi", record=True, timestep=1, clock=recording_clock)
    SM_4 = br.StateMonitor(P, "a", record=True, timestep=1, clock=recording_clock)
    SM_5 = br.StateMonitor(P, "i_s", record=True, timestep=1, clock=recording_clock)
    SMs.append(SM_1)
    SMs.append(SM_2)
    SMs.append(SM_3)
    SMs.append(SM_4)
    SMs.append(SM_5)
   
    if x.tonic_input: 
        network = br.Network(P, Ce, Ci, Ca, M, *SMs)
    else:
        network = br.Network(P, Ps, Ce, Ci, Ca, Cs, M, *SMs)
        
    # starting the simulation
    network.run(x.t_end*br.ms)

    return SMs, M, GM

def simulate_rate4(parameters):

    x = Bunch(parameters)

    GM = generative_model_data(x.Ndic, x.Ndim, x.k, positive=x.positive_GM, seed=x.seed)
    
    tau_m_inv = 1./(x.tau_m*br.ms)

    br.clear(all=True) # clears away all the brian objects from previous simulations.
    
    def thresh_func(values, threshold, thr_type, thr_coeff, nonnegative):
        vals = values.copy()
        
        if not nonnegative:
            vals[(vals<threshold) & (vals>-threshold)]=0.
            if thr_type == "soft":
                vals[(vals>threshold)] -= threshold
                vals[(vals<-threshold)] += threshold
        else:
            vals[(vals<threshold)]=0.
            if thr_type == "soft":
                vals[(vals>threshold)] -= threshold
            
        return thr_coeff * vals
    
    eqs = '''
          du/dt=(-u-li+i_s)*tau_m_inv     :br.Hz
          a                               :br.Hz
          i_s                             :br.Hz
          li                              :br.Hz
          '''
    
    simulation_clock = br.Clock(dt=x.dt*br.ms)
    recording_clock = br.Clock(dt=1*br.ms)
    
    P = br.NeuronGroup(x.Ndic, model=eqs, threshold=200000*br.Hz, clock=simulation_clock)
    P.i_s = x.input_factor * GM["i_stim"]
    P.a = 0
    
    @br.network_operation(clock=simulation_clock, when="end")
    def set_input(simulation_clock):
        P.li = np.dot(GM["G"], P.a)
        P.a = thresh_func(P.u, x.threshold, x.thr_type, x.thr_coeff, x.nonnegative)
        
    SMs = []
    SM_2 = br.StateMonitor(P, "a", record=True, timestep=1, clock=recording_clock)
    SMs.append(SM_2)

    network = br.Network(P, set_input, *SMs)

    # starting the simulation
    network.run(x.t_end*br.ms)

    return 1.*SM_2[:][:,-1]/x.input_factor, GM


def simulate_rate5(parameters):

    x = Bunch(parameters)

    tau_m_inv = 1./(x.tau_m*br.ms)

    br.clear(all=True) # clears away all the brian objects from previous simulations.
   
    assert x.thresh_func, "you have to provide a threshold function"
 
    eqs = '''
          du/dt=(-u-li+i_s)*tau_m_inv     :br.Hz
          a                               :br.Hz
          i_s                             :br.Hz
          li                              :br.Hz
          '''
    
    simulation_clock = br.Clock(dt=x.dt*br.ms)
    recording_clock = br.Clock(dt=1*br.ms)
    
    P = br.NeuronGroup(x.Ndic, model=eqs, threshold=200000*br.Hz, clock=simulation_clock)
    P.i_s = x.input_factor * x.GM["i_stim"]
    P.a = 0
    
    @br.network_operation(clock=simulation_clock, when="end")
    def set_input(simulation_clock):
        P.li = np.dot(x.GM["G"], P.a)
        P.a = x.thresh_func(P.u, x.threshold)
        
    SMs = []
    SM_2 = br.StateMonitor(P, "a", record=True, timestep=1, clock=recording_clock)
    SMs.append(SM_2)

    network = br.Network(P, set_input, *SMs)

    # starting the simulation
    network.run(x.t_end*br.ms)

    return 1.*SM_2[:]
