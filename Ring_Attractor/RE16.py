from brian2 import *
from equations import *
from connections import build_pen_to_epg_indices, build_pen_to_epg_array
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# set_device('cpp_standalone', build_on_run=False)


def visual_cue(theta, index, stimulus=0.03, sigma=2 * np.pi/8):
    """
    Vectorized visual cue function for efficient computation.
    
    Args:
        theta: the angle of the visual input
        index: the index of the neuron (can be array)
        stimulus: the strength of the visual input
        sigma: the standard deviation of the Gaussian distribution
    """
    # Ensure index is numpy array for vectorization
    index = np.asarray(index)
    phi = index * np.pi/8
    
    # Vectorized computation of distances
    d1 = (theta - phi)**2 
    d2 = (theta - phi + 2*np.pi)**2
    d3 = (theta - phi - 2*np.pi)**2
    
    # Vectorized exponential computation
    sigma_sq_2 = 2 * sigma**2
    return stimulus * (np.exp(-d1/sigma_sq_2) + np.exp(-d2/sigma_sq_2) + np.exp(-d3/sigma_sq_2))

def visual_cue_vectorized(theta_r, theta_l, stimulus=0.03, sigma=2 * np.pi/8):
    """
    Compute visual cues for all EPG groups at once.
    
    Args:
        theta_r: right eye angle
        theta_l: left eye angle  
        stimulus: stimulus strength
        sigma: standard deviation
        
    Returns:
        Array of visual cues for all 16 EPG groups
    """
    # Create index arrays for all groups
    indices_r = np.arange(8)  # Right groups 0-7
    indices_l = np.arange(8, 16)  # Left groups 8-15
    
    # Compute cues for all groups at once
    cues_r = visual_cue(theta_r, indices_r, stimulus, sigma)
    cues_l = visual_cue(theta_l, indices_l, stimulus, sigma)
    
    return np.concatenate([cues_r, cues_l])

def simulator( 
        # parameters
        w_EE = 0.719, # EB <-> EB
        w_EI = 0.143, # R -> EB
        w_IE = 0.740, # EB -> R
        w_II = 0.01, # R <-> R
        w_PP = 0.01, # PEN <-> PEN
        w_EP = 0.012, # EB -> PEN 
        w_PE = 0.709, # PEN -> EB
        sigma = 0.0001, # noise level
        events = None,
):
    """Simulate the head direction network with visual cues and body rotation."""
    start_scope()  
    
    taum   = 20*ms   # time constant
    Cm     = 0.1
    g_L    = 10   # leak conductance
    E_l    = -0.07  # leak reversal potential (volt)
    E_e    = 0   # excitatory reversal potential
    tau_e  = 5*ms    # excitatory synaptic time constant
    Vr     = E_l     # reset potential
    Vth    = -0.05  # spike threshold (volt)
    Vs     = 0.02   # spiking potential (volt)
    w_e    = 0.1  	 # excitatory synaptic weight (units of g_L)
    v_e    = 5*Hz    # excitatory Poisson rate
    N_e         = 100     # number of excitatory inputs
    E_ach       = 0
    tau_ach     = 10*ms
    E_GABAA     = -0.07 # GABAA reversal potential
    tau_GABAA   = 5*ms # GABAA synaptic time constant

    # create neuron
    EPG = NeuronGroup(48, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    PEN = NeuronGroup(48,model=eqs_PEN, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron1
    EPG.v = E_l
    PEN.v = E_l
    R.v = E_l

    EPG_groups = [EPG[i:i+3] for i in range(0, 48, 3)]
    PEN_groups = [PEN[i:i+3] for i in range(0, 48, 3)]
    R_groups = [R[0:3]]
    
    # ========= EPG -> EPG =========
    S_EE = Synapses(EPG, EPG, Ach_eqs, on_pre='s_ach += w_EE', method='euler')
    S_EE.connect(condition='i//3 == j//3 and i != j')

    # ========= PEN -> PEN =========
    S_PP = Synapses(PEN, PEN, Ach_eqs_PP, on_pre='s_ach += w_PP', method='euler')
    S_PP.connect(condition='i//3 == j//3 and i != j')
    
    # ========= EPG -> R =========
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    S_EI.connect(condition='True')
    
    # ========= EPG <- R =========
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    S_IE.connect(condition='True')
    
    # ========= R -> R =========
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    # ========= EPG -> PEN =========
    S_EP = Synapses(EPG, PEN, Ach_eqs_EP, on_pre='s_ach += w_EP', method='euler')
    S_EP.connect(condition='i//3 == j//3')
    
    # ========= PEN -> EPG (optimized by connectivity matrix) =========
    S_PE_2 = Synapses(PEN, EPG, model=Ach_eqs_PE_2, on_pre='s_ach += 2*w_PE', method='euler')
    S_PE_1 = Synapses(PEN, EPG, model=Ach_eqs_PE_1, on_pre='s_ach += 1*w_PE', method='euler')

    pre2, post2, pre1, post1 = build_pen_to_epg_array()

    # ---- build all connections at once ----
    S_PE_2.connect(i=pre2, j=post2)
    S_PE_1.connect(i=pre1, j=post1)
    # ========= end PEN -> EPG =========
    # record model state

    # Create monitors efficiently using list comprehensions
    PRM_EPG = [PopulationRateMonitor(group) for group in EPG_groups]
    PRM_PEN = [PopulationRateMonitor(group) for group in PEN_groups]
    PRM_R = [PopulationRateMonitor(group) for group in R_groups]
    
    net=Network(collect())
    net.add(S_EP,S_EE,S_PP,S_EI,S_IE,S_II)
    # Explicitly add monitors to the network
    for monitors in [PRM_EPG, PRM_PEN, PRM_R]:
        net.add(*monitors)

    # run simulation

    ## SIMULATION ###

    def visual_cues_on(location, strength):
        location %= 2*np.pi
        theta_r = location/2
        theta_l = theta_r + np.pi
        visual_cues = visual_cue_vectorized(theta_r, theta_l, strength)
        
        # Apply visual cues to all EPG groups at once
        for i, cue in enumerate(visual_cues):
            EPG_groups[i].I = cue

    def visual_cues_off():
        EPG.I = 0

    def right(strength):
        PEN.I = 0
        for i in range(8):
            PEN_groups[i].I = strength
    def left(strength):
        PEN.I = 0
        for i in range(8,16):
            PEN_groups[i].I = strength

    # Event-driven stimulation sequence 
    if events is None:
        events = [
            {'type': 'visual_cue_on', 'location': 0, 'strength': 0.05, 'duration': 300*ms},
            {'type': 'visual_cue_off', 'duration': 300*ms},
            {'type': 'shift', 'direction': 'right', 'strength': 0.015, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'left', 'strength': 0.015, 'duration': 1000*ms},
        ]

    for ev in events:
        etype = ev.get('type')  
        duration = ev.get('duration', None)

        if etype == 'visual_cue_on':
            location = ev.get('location')
            strength = ev.get('strength')
            visual_cues_on(location, strength)
        elif etype == 'visual_cue_off':
            visual_cues_off()
        elif etype == 'shift':
            direction = ev.get('direction')
            strength = ev.get('strength')
            if direction == 'right':
                right(strength)
            elif direction == 'left':
                left(strength)
        elif etype == 'wait' or etype == 'run':
            pass
        else:
            raise ValueError(f'Unknown event type: {etype}')

        if duration is not None:
            net.run(duration)


    end  = time.time()

    smooth_width = 5*ms
    fr_epg = np.array([prm.smooth_rate(width=smooth_width) for prm in PRM_EPG])
    fr_pen = np.array([prm.smooth_rate(width=smooth_width) for prm in PRM_PEN])
    fr_r = np.array([prm.smooth_rate(width=smooth_width) for prm in PRM_R])
    t = np.linspace(0, len(fr_epg[0])/10000, len(fr_epg[0]))
    return t, fr_epg, fr_pen, fr_r

if __name__ == '__main__':
    t, fr, fr_pen, fr_r = simulator()    
