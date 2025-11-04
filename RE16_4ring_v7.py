from brian2 import *
from RE16_4ring_v4_eqs import *
from RE16_4ring_v3_con import build_pen_to_epg_array
import numpy as np
import time
from tqdm import tqdm, trange

TOTAL_NEURONS = 12288
TOTAL_INDEX = 4096
TN = 12288
TI = 4096

def map_index(i):
    g = i // 3  # subgroup index
    o = i % 3   # offset in subgroup
    return g * (16*3) + o

def map_index_PENz(i):
    g = i // 3  # subgroup index
    o = i % 3   # offset in subgroup
    return g * (256*3) + o

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

    start = time.time()
    # Device and performance preferences (must be set before creating objects)
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
    E_l    = -0.07  # leak reversal potential (volt)

    total = 16
    class _NoOpPbar:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def update(self, n=1):
            return None
        def set_description(self, *_args, **_kwargs):
            return None
    pbar_cm = tqdm(total=total, desc="Creating neurons") if progress else _NoOpPbar()
    with pbar_cm as pbar:
        EPG = NeuronGroup(TOTAL_NEURONS, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
        PENx = NeuronGroup(TOTAL_NEURONS,model=eqs_PENx, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
        PENy = NeuronGroup(TOTAL_NEURONS,model=eqs_PENy, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
        PENz = NeuronGroup(TOTAL_NEURONS,model=eqs_PENz, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
        R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
        pbar.update(1)
        pbar.set_description("Initializing neurons")
        EPG.v = E_l
        PENx.v = E_l
        PENy.v = E_l
        PENz.v = E_l
        R.v = E_l
        pbar.update(1)
        pbar.set_description("Grouping neurons")

        EPG_groups  = [[[EPG[((i*256)+(j*16)+k)*3:((i*256)+(j*16)+k)*3+3] for k in range(16)] for j in range(16)] for i in range(16)]
        PENx_groups = [[[PENx[((i*256)+(j*16)+k)*3:((i*256)+(j*16)+k)*3+3] for k in range(16)] for j in range(16)] for i in range(16)]
        PENy_groups = [[[PENy[((i*256)+(j*16)+k)*3:((i*256)+(j*16)+k)*3+3] for k in range(16)] for j in range(16)] for i in range(16)]
        PENz_groups = [[[PENz[((i*256)+(j*16)+k)*3:((i*256)+(j*16)+k)*3+3] for k in range(16)] for j in range(16)] for i in range(16)]
        R_groups = [R[0:3]]
        pbar.update(1)

        # ========= EPG -> EPG =========
        pbar.set_description("Building EPG -> EPG connections")
        S_EE = Synapses(EPG, EPG, Ach_eqs_EE, on_pre='s_ach += w_EE', method='euler')
        S_EE.connect(condition='i//3 == j//3 and i != j')
        pbar.update(1)
        # ========= PEN -> PEN =========
        pbar.set_description("Building PEN -> PEN connections")
        S_PPx = Synapses(PENx, PENx, Ach_eqs_PPx, on_pre='s_ach += w_PP', method='euler')
        S_PPx.connect(condition='i//3 == j//3 and i != j')
        S_PPy = Synapses(PENy, PENy, Ach_eqs_PPy, on_pre='s_ach += w_PP', method='euler')
        S_PPy.connect(condition='i//3 == j//3 and i != j')
        S_PPz = Synapses(PENz, PENz, Ach_eqs_PPz, on_pre='s_ach += w_PP', method='euler')
        S_PPz.connect(condition='i//3 == j//3 and i != j')
        pbar.update(1)
        # ========= EPG -> R =========
        pbar.set_description("Building EPG -> R connections")
        S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
        S_EI.connect(condition='True')
        pbar.update(1)
        # ========= R   -> EPG =========
        pbar.set_description("Building R -> EPG connections")
        S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
        S_IE.connect(condition='True')
        pbar.update(1)
        # ========= R <-> R =========
        pbar.set_description("Building R <-> R connections")
        S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
        S_II.connect(condition='i != j')
        pbar.update(1)
        # ========= EPG -> PEN =========
        pbar.set_description("Building EPG -> PEN connections")
        S_EPx = Synapses(EPG, PENx, Ach_eqs_EPx, on_pre='s_ach += w_EP', method='euler')
        S_EPx.connect(condition='i//3 == j//3')
        S_EPy = Synapses(EPG, PENy, Ach_eqs_EPy, on_pre='s_ach += w_EP', method='euler')
        S_EPy.connect(condition='i//3 == j//3')   
        S_EPz = Synapses(EPG, PENz, Ach_eqs_EPz, on_pre='s_ach += w_EP', method='euler')
        S_EPz.connect(condition='i//3 == j//3')
        pbar.update(1)
        # ========= PEN -> EPG (vectorized connectivity) =========
        pbar.set_description("Building PEN -> EPG connections")
        S_PxE_2 = Synapses(PENx, EPG, model=Ach_eqs_PxE_2, on_pre='s_ach += 2*w_PE', method='euler')
        S_PxE_1 = Synapses(PENx, EPG, model=Ach_eqs_PxE_1, on_pre='s_ach += 1*w_PE', method='euler')
        S_PyE_2 = Synapses(PENy, EPG, model=Ach_eqs_PyE_2, on_pre='s_ach += 2*w_PE', method='euler')
        S_PyE_1 = Synapses(PENy, EPG, model=Ach_eqs_PyE_1, on_pre='s_ach += 1*w_PE', method='euler')
        S_PzE_2 = Synapses(PENz, EPG, model=Ach_eqs_PzE_2, on_pre='s_ach += 2*w_PE', method='euler')
        S_PzE_1 = Synapses(PENz, EPG, model=Ach_eqs_PzE_1, on_pre='s_ach += 1*w_PE', method='euler')
        pbar.update(1)
        pre2, post2, pre1, post1 = build_pen_to_epg_array()

        for j in range(16):
            for k in range(1):
                S_PxE_2.connect(i=pre2+j*48+k*256*3, j=post2+j*48+k*256*3)
                S_PxE_1.connect(i=pre1+j*48+k*256*3, j=post1+j*48+k*256*3)
                S_PyE_2.connect(i=map_index(pre2)+j*3+k*256*3, j=map_index(post2)+j*3+k*256*3)
                S_PyE_1.connect(i=map_index(pre1)+j*3+k*256*3, j=map_index(post1)+j*3+k*256*3)
                S_PzE_2.connect(i=map_index_PENz(pre2)+(j+16*k)*3, j=map_index_PENz(post2)+(j+16*k)*3)
                S_PzE_1.connect(i=map_index_PENz(pre1)+(j+16*k)*3, j=map_index_PENz(post1)+(j+16*k)*3)
        # ========= end PEN -> EPG =========

        # record model state
        pbar.set_description("Recording model state")
        PRM_EPG = [PopulationRateMonitor(EPG_groups[i][j][k]) for i in range(16) for j in range(16) for k in range(16)]
        PRM_PENx = [PopulationRateMonitor(PENx_groups[i][j][k]) for i in range(16) for j in range(16) for k in range(16)]
        PRM_PENy = [PopulationRateMonitor(PENy_groups[i][j][k]) for i in range(16) for j in range(16) for k in range(16)]
        PRM_PENz = [PopulationRateMonitor(PENz_groups[i][j][k]) for i in range(16) for j in range(16) for k in range(16)]

        pbar.update(1)

        pbar.set_description("Collecting network")
        net = Network(collect())
        net.add(S_EPx,S_EPy,S_EE,S_PPx,S_PPy,S_PPz,S_EI,S_IE,S_II,S_PxE_2,S_PxE_1,S_PyE_2,S_PyE_1,S_PzE_2,S_PzE_1)
        for monitors in [PRM_EPG, PRM_PENx, PRM_PENy, PRM_PENz]:
            net.add(*monitors)
        pbar.update(1)
        

        ## SIMULATION ###
        print(f'\r{time.strftime("%H:%M:%S")} : {(time.time() - start)//60:.0f} min {(time.time() - start)%60:.1f} sec -> simulation start')
        pbar.set_description("Running simulation")
        pbar.update(1)

        def visual_cue_on(x, y, z):
            pbar.set_description("Visual cue on")            
            d = 8
            EPG_groups[(x)%16][(y)%16][(z)%16].I = 0.5
            EPG_groups[(x+d)%16][(y)%16][(z)%16].I = 0.5
            EPG_groups[(x)%16][(y+d)%16][(z)%16].I = 0.5
            EPG_groups[(x)%16][(y)%16][(z+d)%16].I = 0.5
            EPG_groups[(x+d)%16][(y+d)%16][(z)%16].I = 0.5
            EPG_groups[(x+d)%16][(y)%16][(z+d)%16].I = 0.5
            EPG_groups[(x)%16][(y+d)%16][(z+d)%16].I = 0.5
            EPG_groups[(x+d)%16][(y+d)%16][(z+d)%16].I = 0.5


        def visual_cue_off():
            pbar.set_description("Visual cue off")
            EPG.I = 0

        def reset():
            PENx.I = 0
            PENy.I = 0
            PENz.I = 0

        def right(strength=0.015):
            pbar.set_description("Right")
            reset()
            for i in range(8):
                for j in range(16):
                    for k in range(1):
                        PENx_groups[k][j][i].I = strength

        def left(strength):
            pbar.set_description("Left")
            reset()
            for i in range(8, 16):
                for j in range(16):
                    for k in range(16):
                        PENx_groups[k][j][i].I = strength

        def up(strength):
            pbar.set_description("Up")
            reset()
            for i in range(16):
                for j in range(8):
                    for k in range(16):
                        PENy_groups[k][j][i].I = strength

        def down(strength):
            pbar.set_description("Down")
            reset()
            for i in range(16):
                for j in range(8, 16):
                    for k in range(16):
                        PENy_groups[k][j][i].I = strength

        def front(strength):
            pbar.set_description("Front")
            reset()
            for i in range(16):
                for j in range(16):
                    for k in range(8):
                        PENz_groups[k][j][i].I = strength

        def back(strength):
            pbar.set_description("Back")
            reset()
            for i in range(16):
                for j in range(16):
                    for k in range(8, 16):
                        PENz_groups[k][j][i].I = strength

        
        
        visual_cue_on(1, 2, 3)
        net.run(300 * ms)
        pbar.update(1)

        visual_cue_off()
        net.run(300 * ms)
        pbar.update(1)

        right(shifter_strength)
        net.run(1000 * ms)
        pbar.update(1)

        end  = time.time()
        print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> simulation end', flush=True)

    fr = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_EPG])
    fr_penx = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_PENx])
    fr_peny = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_PENy])
    fr_penz = np.array([prm.smooth_rate(width=5*ms) for prm in PRM_PENz])

    t = np.linspace(0, len(fr[0])/(1000*defaultclock_dt), len(fr[0]))
    print(fr.shape)
    time_length = fr.shape[1]
    fr = fr.reshape(16, 16, 16, time_length)
    print(f"Reshaped fr to: {fr.shape}")  # Should print (16, 16, 16, time_length)
    return t, fr, fr_penx, fr_peny, fr_penz

if __name__ == '__main__':
    t, fr, fr_penx, fr_peny, fr_penz = simulator(device_mode='runtime', codegen_target='numpy', progress=False)