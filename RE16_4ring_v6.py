from brian2 import *
from RE16_4ring_v4_eqs import *
from RE16_4ring_v3_con import build_pen_to_epg_indices, build_pen_to_epg_array
import numpy as np
import time
from tqdm import tqdm, trange


def visual_cue(theta, index, stimulus = 0.03, sigma = 0.8 * np.pi/8):
    """
    param: 
    theta: the angle of the visual input
    index: the index of the neuron
    stimulus: the strength of the visual input
    sigma: the standard deviation of the Gaussian distribution
    """
    A = stimulus
    phi = (index * np.pi/8) % (2*np.pi)
    d1 = (theta-phi)**2 
    d2 = (theta-phi + 2*np.pi)**2
    d3 = (theta-phi - 2*np.pi)**2
    return A * (np.exp(-d1/(2*sigma**2)) + np.exp(-d2/(2*sigma**2)) + np.exp(-d3/(2*sigma**2)))

def visual_cue_2D(x, y, i, j, stimulus = 0.03, sigma = 0.8 * np.pi/8):

    A = stimulus
    r1 = np.sqrt((x-i)**2 + (y-j)**2) 

    return A * (np.exp(-r1**2/(2*sigma**2)))



def map_index(i):
    g = i // 3  # subgroup index
    o = i % 3   # offset in subgroup
    return g * 48 + o


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
        
        stimulus_strength = 0.5, 
        stimulus_location =0 , # from 0 to np.pi
        shifter_strength = 0.015,

        # debug
        debug = False,
        target = 0,

        # performance settings
        device_mode = 'runtime', # 'runtime' | 'cpp_standalone' | 'cuda_standalone'
        build_dir = 'brian_standalone',
        use_cython = False,
        codegen_target = 'numpy', # 'numpy' | 'cython'
        use_float32 = False,
        defaultclock_dt = 0.1*ms,
        progress = True,
):
    """Simulate the head direction network with visual cues and body rotation."""

    if debug:
        print(f'{time.strftime("%H:%M:%S")} [info] Parameters:')
        print(f'w_EE: {w_EE}')
        print(f'w_EI: {w_EI}')
        print(f'w_IE: {w_IE}')
        print(f'w_II: {w_II}')
        print(f'w_PP: {w_PP}')
        print(f'w_EP: {w_EP}')
        print(f'w_PE: {w_PE}')
        print(f'sigma: {sigma}')
        print(f'stimulus_strength: {stimulus_strength}')
        print(f'stimulus_location: {stimulus_location}')
        print(f'shifter_strength: {shifter_strength}')

    start = time.time()
    if device_mode in ('cpp_standalone', 'cuda_standalone'):
        set_device(device_mode, directory=build_dir, build_on_run=True)
    else:
        # Explicitly set code generation target for runtime execution
        try:
            if use_cython:
                prefs.codegen.target = 'cython'
            elif codegen_target in ('numpy', 'cython'):
                prefs.codegen.target = codegen_target
        except Exception:
            pass
    if use_float32:
        prefs.core.default_float_dtype = float32
    defaultclock.dt = defaultclock_dt


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
    target = target
        

    EPG = NeuronGroup(16*16*3, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    PENx = NeuronGroup(16*16*3,model=eqs_PENx, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    PENy = NeuronGroup(16*16*3,model=eqs_PENy, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron
    EPG.v = E_l
    PENx.v = E_l
    PENy.v = E_l
    R.v = E_l

    EPG_groups = [EPG[i:i+3] for i in range(0, 16*16*3, 3)]
    PENx_groups = [PENx[i:i+3] for i in range(0, 16*16*3, 3)]
    PENy_groups = [PENy[i:i+3] for i in range(0, 16*16*3, 3)]
    R_groups = [R[0:3]]
    
    # ========= EPG -> EPG =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building EPG -> EPG connections')

    S_EE = Synapses(EPG, EPG, Ach_eqs_EE, on_pre='s_ach += w_EE', method='euler')
    S_EE.connect(condition='i//3 == j//3 and i != j')

    # ========= PEN -> PEN =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building PEN -> PEN connections')
    S_PPx = Synapses(PENx, PENx, Ach_eqs_PPx, on_pre='s_ach += w_PP', method='euler')
    S_PPx.connect(condition='i//3 == j//3 and i != j')
    S_PPy = Synapses(PENy, PENy, Ach_eqs_PPy, on_pre='s_ach += w_PP', method='euler')
    S_PPy.connect(condition='i//3 == j//3 and i != j')
    
    # ========= EPG -> R =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building EPG -> R connections')
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    S_EI.connect(condition='True')
    
    # ========= R   -> EPG =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building R -> EPG connections')
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    S_IE.connect(condition='True')
    
    # ========= R <-> R =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building R <-> R connections')
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    # ========= EPG -> PEN =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building EPG -> PEN connections')
    S_EPx = Synapses(EPG, PENx, Ach_eqs_EPx, on_pre='s_ach += w_EP', method='euler')
    S_EPx.connect(condition='i//3 == j//3')
    S_EPy = Synapses(EPG, PENy, Ach_eqs_EPy, on_pre='s_ach += w_EP', method='euler')
    S_EPy.connect(condition='i//3 == j//3')

    # ========= PEN -> EPG (optimized by connectivity matrix) =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building PEN -> EPG connections')
    S_PxE_2 = Synapses(PENx, EPG, model=Ach_eqs_PxE_2, on_pre='s_ach += 2*w_PE', method='euler')
    S_PxE_1 = Synapses(PENx, EPG, model=Ach_eqs_PxE_1, on_pre='s_ach += 1*w_PE', method='euler')
    S_PyE_2 = Synapses(PENy, EPG, model=Ach_eqs_PyE_2, on_pre='s_ach += 2*w_PE', method='euler')
    S_PyE_1 = Synapses(PENy, EPG, model=Ach_eqs_PyE_1, on_pre='s_ach += 1*w_PE', method='euler')

    pre2, post2, pre1, post1 = build_pen_to_epg_array()

    # # ---- build all connections at once ----
    for k in range(16):
        S_PxE_2.connect(i=pre2+k*48, j=post2+k*48)
        S_PxE_1.connect(i=pre1+k*48, j=post1+k*48)
        S_PyE_2.connect(i=map_index(pre2)+k*3, j=map_index(post2)+k*3)
        S_PyE_1.connect(i=map_index(pre1)+k*3, j=map_index(post1)+k*3)

    # ========= end PEN -> EPG =========

    # record model state
    PRM_EPG = [PopulationRateMonitor(group) for group in EPG_groups]
    PRM_PENx = [PopulationRateMonitor(group) for group in PENx_groups]
    PRM_PENy = [PopulationRateMonitor(group) for group in PENy_groups]
    PRM_R = [PopulationRateMonitor(group) for group in R_groups]    
    
    net = Network(collect())
    net.add(S_EPx,S_EPy,S_EE,S_PPx,S_PPy,S_EI,S_IE,S_II,S_PxE_2,S_PxE_1,S_PyE_2,S_PyE_1)
    # Explicitly add monitors to the network
    for monitors in [PRM_EPG, PRM_PENx, PRM_PENy, PRM_R]:
        net.add(*monitors)
    # run simulation

    ## SIMULATION ###
    print(f'\r{time.strftime("%H:%M:%S")} : {(time.time() - start)//60:.0f} min {(time.time() - start)%60:.1f} sec -> simulation start', flush=True)

    # DTheta / Dt = omega
    # DTheta = omega * Dt
    # Dx range: 0 to 2*pi
    w = 10
    Dx = w * 0.1
    Dy = 1
    Dt = 100 # ms
    A = stimulus_strength
    theta_r = stimulus_location/2
    theta_l = theta_r + np.pi

        

    for i in range(target*16,target*16+8):
        EPG_groups[i%256].I = visual_cue(theta_r, i, 0.05)
        EPG_groups[(i+8)%256].I = visual_cue(theta_l, i+8, 0.05)
        EPG_groups[(i+128)%256].I = visual_cue(theta_r, i+128, 0.05)
        EPG_groups[(i+136)%256].I = visual_cue(theta_l, i+136, 0.05)
    net.run(300 * ms)

    EPG.I = 0
    net.run(300 * ms)

    
    def reset():
        for i in range(256):
            PENx_groups[i].I = 0
            PENy_groups[i].I = 0

    def right(strength):
        for i in range(8): 
            for j in range(16):
                PENx_groups[i+j*16].I = strength
    def left(strength):
        for i in range(8,16): 
            for j in range(16):
                PENx_groups[i+j*16].I = strength
    def up(strength):
        for i in range(8):
            for j in range(16):
                PENy_groups[i*16+j].I = strength
    def down(strength):
        for i in range(8,16):
            for j in range(16):
                PENy_groups[i*16+j].I = strength

    right(shifter_strength)
    net.run(1000 * ms)
    end  = time.time()

    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> simulation end', flush=True)

    fr = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_EPG])
    fr_penx = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_PENx])
    fr_peny = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_PENy])
    fr_r = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_R])
    t = np.linspace(0, len(fr[0])/10000, len(fr[0]))
    print(fr.shape)
    time_length = fr.shape[1]
    fr = fr.reshape(16, 16, time_length)
    print(f"Reshaped fr to: {fr.shape}")  # Should print (16, 16, time_length)

    return t, fr, fr_r, fr_pen

if __name__ == '__main__':
    t, fr, fr_r, fr_pen = simulator()    