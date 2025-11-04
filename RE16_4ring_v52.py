from brian2 import *
from RE16_4ring_v4_eqs import *
from RE16_4ring_v3_con import build_pen_to_epg_array
import numpy as np
import time
import brian2 as b2

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

        defaultclock_dt = 0.1*ms,
        events=None,
        seed=830,
):
    """Simulate the head direction network with visual cues and body rotation."""

    start = time.time()
    if hasattr(defaultclock_dt, 'dim'):
        defaultclock.dt = defaultclock_dt
    else:
        defaultclock.dt = defaultclock_dt * ms
    start_scope()  
    
    E_l    = -0.07  # leak reversal potential (volt)
    # create neuron
    EPG = NeuronGroup(256*3, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    PENx = NeuronGroup(256*3,model=eqs_PENx, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    PENy = NeuronGroup(256*3,model=eqs_PENy, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    b2.seed(seed)

    # initialize neuron
    EPG.v = E_l
    PENx.v = E_l
    PENy.v = E_l
    R.v = E_l

    # EPG_groups = [EPG[i:i+3] for i in range(0, 256*3, 3)]
    EPG_groups = [[EPG[i*48 + j*3:i*48 + j*3 + 3] for j in range(16)] for i in range(16)]
    PENx_groups = [[PENx[i*48 + j*3:i*48 + j*3 + 3] for j in range(16)] for i in range(16)]
    PENy_groups = [[PENy[i*48 + j*3:i*48 + j*3 + 3] for j in range(16)] for i in range(16)]

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
    PRM_EPG = [PopulationRateMonitor(EPG_groups[i][j]) for i in range(16) for j in range(16)]
    PRM_PENx = [PopulationRateMonitor(PENx_groups[i][j]) for i in range(16) for j in range(16)]
    PRM_PENy = [PopulationRateMonitor(PENy_groups[i][j]) for i in range(16) for j in range(16)]

    net = Network(collect())
    net.add(S_EPx,S_EPy,S_EE,S_PPx,S_PPy,S_EI,S_IE,S_II,S_PxE_2,S_PxE_1,S_PyE_2,S_PyE_1)
    net.add(*PRM_EPG)
    net.add(*PRM_PENx)
    net.add(*PRM_PENy)
    
    def reset():
        PENx.I = 0
        PENy.I = 0

    def right(strength=0.015):
        reset()
        for i in range(16):
            for j in range(8):
                PENx_groups[i][j].I = strength

    def left(strength):
        reset()
        for i in range(16):
            for j in range(8, 16):
                PENx_groups[i][j].I = strength

    def up(strength):
        reset()
        for i in range(8):
            for j in range(16):
                PENy_groups[i][j].I = strength

    def down(strength):
        reset()
        for i in range(8, 16):
            for j in range(16):
                PENy_groups[i][j].I = strength

    def visual_cue_on(x, y, strength=0.05):
        EPG_groups[x][y].I = strength
        EPG_groups[x][y+8].I = strength
        EPG_groups[x+8][y].I = strength
        EPG_groups[x+8][y+8].I = strength
        
    def visual_cue_off():
        EPG.I = 0

    ## SIMULATION ###
    print(f'\r{time.strftime("%H:%M:%S")} : {(time.time() - start)//60:.0f} min {(time.time() - start)%60:.1f} sec -> simulation start', flush=True)

    def _to_b2_time(value):
        return value if hasattr(value, 'dim') else value*ms

    def apply_shift(direction, strength):
        mapping = {'right': right, 'left': left, 'up': up, 'down': down}
        try:
            mapping[direction](strength)
        except KeyError:
            raise ValueError(f'Unknown shift direction: {direction}')

    # Event-driven stimulation sequence
    if events is None:
        shifter_strength = 0.018
        events = [
            {'type': 'visual_cue_on', 'x': 2, 'y': 6, 'strength': 0.05, 'duration': 300*ms},
            {'type': 'visual_cue_off', 'duration': 300*ms},
            {'type': 'shift', 'direction': 'right', 'strength': shifter_strength, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'up', 'strength': shifter_strength, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'left', 'strength': shifter_strength, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'down', 'strength': shifter_strength, 'duration': 1000*ms},
        ]

    for ev in events:
        etype = ev.get('type')
        duration = ev.get('duration', None)

        if etype == 'visual_cue_on':
            visual_cue_on(ev['x'], ev['y'], ev['strength'])
        elif etype == 'visual_cue_off':
            visual_cue_off()
        elif etype == 'shift':
            direction = ev['direction']
            strength = ev['strength']
            apply_shift(direction, strength)
        elif etype == 'wait' or etype == 'run':
            pass
        else:
            raise ValueError(f'Unknown event type: {etype}')

        if duration is not None:
            net.run(duration)

    end  = time.time()

    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> simulation end', flush=True)
    smooth_width = 5*ms
    fr_epg = np.array([prm.smooth_rate(width=smooth_width) for prm in PRM_EPG])
    fr_penx = np.array([prm.smooth_rate(width=smooth_width) for prm in PRM_PENx])
    fr_peny = np.array([prm.smooth_rate(width=smooth_width) for prm in PRM_PENy])
    # time vector from monitors (in ms)
    t = np.asarray(PRM_EPG[0].t/ms)
    time_length = fr_epg.shape[1]
    fr = fr_epg.reshape(16, 16, time_length)
    fr_penx = fr_penx.reshape(16, 16, time_length)
    fr_peny = fr_peny.reshape(16, 16, time_length)
    return t, fr, fr_penx, fr_peny

if __name__ == '__main__':
    from util import process_data
    import matplotlib.pyplot as plt
    shifter_strength = 0.018
    events = [
            {'type': 'visual_cue_on', 'x': 2, 'y': 6, 'strength': 0.5, 'duration': 300*ms},
            {'type': 'visual_cue_off', 'duration': 300*ms},
            {'type': 'shift', 'direction': 'right', 'strength': shifter_strength, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'up', 'strength': shifter_strength, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'left', 'strength': shifter_strength, 'duration': 1000*ms},
            {'type': 'shift', 'direction': 'down', 'strength': shifter_strength, 'duration': 1000*ms},
        ]
    t, fr = simulator(shifter_strength = shifter_strength, events = events)    
    data = process_data(fr)
    plt.figure(figsize=(8, 6))
    last_frame = data[:, :, -1]
    im_last = plt.imshow(last_frame, cmap='Blues', aspect='auto', interpolation='none',
                        vmin=0, vmax=200)
    plt.colorbar(im_last, label='Firing Rate [Hz]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Final state (t = {t[-1]/1000:.2f} s)')
    plt.tight_layout()
    plt.show()
    plt.savefig('figures/RE16_grid_v52.png')
    plt.close()