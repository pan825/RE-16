from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import time 
import sys

start = time.time()
print("hello")

start_scope()
			
# parameters
taum   = 20*ms   # time constant
Cm = 0.1
g_L    = 10   # leak conductance
E_l    = -0.07  # leak reversal potential (volt)
E_e    = 0   # excitatory reversal potential
tau_e  = 5*ms    # excitatory synaptic time constant
Vr     = E_l     # reset potential
Vth    = -0.05  # spike threshold (volt)
Vs     = 0.02   # spiking potential (volt)
w_e    = 0.1  	 # excitatory synaptic weight (units of g_L)
v_e    = 5*Hz    # excitatory Poisson rate
N_e    = 100     # number of excitatory inputs

E_ach = 0
tau_ach = 10*ms


E_GABAA  = -0.07 # GABAA reversal potential
tau_GABAA = 5*ms # GABAA synaptic time constant


sigma = 0.001 # noise level


w_EE = 0.49 # EB <-> EB
w_IE = 0.14 # R -> EB
w_EI = 0.10 # EB -> R
w_II = 0 # R <-> R
w_PP = 0 # PEN <-> PEN
w_EP = 0.0 # EB -> PEN
w_PE = 0.0 # PEN -> EB

# model equations
eqs_EPG = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PE + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PE2R :1
Isyn_PE2L :1
Isyn_PE1R :1
Isyn_PE1L :1
Isyn_PE2R2:1
Isyn_PE2L2 :1
Isyn_PE1R2 :1
Isyn_PE1L2:1
Isyn_PE7:1
Isyn_PE8:1
Isyn_PE = Isyn_PE2R + Isyn_PE2L + Isyn_PE1R + Isyn_PE1L + Isyn_PE2R2 + Isyn_PE2L2 + Isyn_PE1R2 + Isyn_PE1L2+Isyn_PE7+Isyn_PE8:1

'''
eqs_R = '''
dv/dt = (IsynEI + Isyn_ii + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
IsynEI : 1
Isyn_ii:1 
'''
eqs_PEN = '''
dv/dt = (Isyn_pp + Isyn_EP + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_pp : 1
Isyn_EP : 1
'''

Ach_eqs = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PP = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_pp_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EP = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EP_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2R = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE2R2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE7 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE7_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE8 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE8_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EI = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynEI_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

GABA_eqs = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_i_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

GABA_eqs_i = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_ii_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''
#dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)

# create neuron
EPG = NeuronGroup(48, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
PEN = NeuronGroup(48,model=eqs_PEN, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
# initialize neuron1
EPG.v = E_l
PEN.v = E_l
R.v = E_l

EPG_groups = []
EPG_groups.append(EPG[0:3])
EPG_groups.append(EPG[3:6])
EPG_groups.append(EPG[6:9])
EPG_groups.append(EPG[9:12])
EPG_groups.append(EPG[12:15])
EPG_groups.append(EPG[15:18])
EPG_groups.append(EPG[18:21])
EPG_groups.append(EPG[21:24])
EPG_groups.append(EPG[24:27])
EPG_groups.append(EPG[27:30])
EPG_groups.append(EPG[30:33])
EPG_groups.append(EPG[33:36])
EPG_groups.append(EPG[36:39])
EPG_groups.append(EPG[39:42])
EPG_groups.append(EPG[42:45])
EPG_groups.append(EPG[45:48])


PEN_groups = []
PEN_groups.append(PEN[0:3])
PEN_groups.append(PEN[3:6])
PEN_groups.append(PEN[6:9])
PEN_groups.append(PEN[9:12])
PEN_groups.append(PEN[12:15])
PEN_groups.append(PEN[15:18])
PEN_groups.append(PEN[18:21])
PEN_groups.append(PEN[21:24])
PEN_groups.append(PEN[24:27])
PEN_groups.append(PEN[27:30])
PEN_groups.append(PEN[30:33])
PEN_groups.append(PEN[33:36])
PEN_groups.append(PEN[36:39])
PEN_groups.append(PEN[39:42])
PEN_groups.append(PEN[42:45])
PEN_groups.append(PEN[45:48])

EPG_syn = []
PEN_syn = []
PE2R_syn = []
PE2L_syn = []
PE1R_syn = []
PE1L_syn = []
PE2R_syn2 = []
PE2L_syn2 = []
PE1R_syn2 = []
PE1L_syn2 = []

EP_syn = []
# create connections
for k in range(0,16):
    EPG_syn.append(Synapses(EPG_groups[k],EPG_groups[k],Ach_eqs,on_pre = 's_ach += w_EE', method ='euler'))
    EPG_syn[k].connect(condition = 'i!=j')
#PEN_PEN
for k2 in range(0,16):
    PEN_syn.append(Synapses(PEN_groups[k2],PEN_groups[k2],Ach_eqs_PP,on_pre = 's_ach += w_PP', method ='euler'))
    PEN_syn[k2].connect(condition = 'i!=j')


S_EI = Synapses(EPG,R,model = Ach_eqs_EI , on_pre='s_ach += w_EI', method = 'euler')
for a in range(0,54):
    for b in range(0,3):
        S_EI.connect(i=a,j=b)

S_IE = Synapses(R,EPG,model = GABA_eqs , on_pre='s_GABAA += w_IE', method = 'euler')
for a2 in range(0,54):
    for b2 in range(0,3):
        S_IE.connect(i=b2,j=a2)

S_II = Synapses(R,R,model = GABA_eqs_i , on_pre='s_GABAA += w_II', method = 'euler')
S_II.connect(condition = 'i!=j')

#EPG_PEN synapse
for k3 in range(0,16):
    EP_syn.append(Synapses(EPG_groups[k3],PEN_groups[k3] ,Ach_eqs_EP,on_pre='s_ach += w_EP', method = 'euler'))
    EP_syn[k3].connect(j='u for u in range(0,3)', skip_if_invalid=True)


#PEN_EPG synapse
#PEN0-6->EPG0-8
for k4 in range(0,7):
    PE2R_syn.append(Synapses(PEN_groups[k4],EPG_groups[k4+1],Ach_eqs_PE2R,on_pre='s_ach += 2*w_PE', method = 'euler'))
    PE2R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
for k4 in range(0,6):
    PE1R_syn.append(Synapses(PEN_groups[k4],EPG_groups[k4+2],Ach_eqs_PE1R,on_pre='s_ach += 1*w_PE', method = 'euler'))
    PE1R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)


#PEN9-15->EPG0-8
for k4 in range(0,7):
    PE2R_syn2.append(Synapses(PEN_groups[k4+9],EPG_groups[k4+1],Ach_eqs_PE2R2,on_pre='s_ach += 2*w_PE', method = 'euler'))
    PE2R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
for k4 in range(0,6):
    PE1R_syn2.append(Synapses(PEN_groups[k4+9],EPG_groups[k4+2],Ach_eqs_PE1R2,on_pre='s_ach += 1*w_PE', method = 'euler'))
    PE1R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)

#PEN7-8
PE7_syn  = []
PE7_syn.append(Synapses(PEN_groups[7],EPG_groups[0],Ach_eqs_PE7,on_pre='s_ach += 3*w_PE', method = 'euler'))
PE7_syn.append(Synapses(PEN_groups[7],EPG_groups[1],Ach_eqs_PE7,on_pre='s_ach += 1*w_PE', method = 'euler'))
PE7_syn.append(Synapses(PEN_groups[7],EPG_groups[15],Ach_eqs_PE7,on_pre='s_ach += 3*w_PE', method = 'euler'))
PE7_syn.append(Synapses(PEN_groups[7],EPG_groups[14],Ach_eqs_PE7,on_pre='s_ach += 1*w_PE', method = 'euler'))
for k in range(0,4):
    PE7_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)

PE8_syn  = []
PE8_syn.append(Synapses(PEN_groups[8],EPG_groups[0],Ach_eqs_PE8,on_pre='s_ach += 3*w_PE', method = 'euler'))
PE8_syn.append(Synapses(PEN_groups[8],EPG_groups[1],Ach_eqs_PE8,on_pre='s_ach += 1*w_PE', method = 'euler'))
PE8_syn.append(Synapses(PEN_groups[8],EPG_groups[15],Ach_eqs_PE8,on_pre='s_ach += 3*w_PE', method = 'euler'))
PE8_syn.append(Synapses(PEN_groups[8],EPG_groups[14],Ach_eqs_PE8,on_pre='s_ach += 1*w_PE', method = 'euler'))
for k in range(0,4):
    PE8_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)


#PEN0-6 ->EPG8-15
for k4 in range(0,7):
    PE2L_syn.append(Synapses(PEN_groups[k4],EPG_groups[k4+8],Ach_eqs_PE2L,on_pre='s_ach += 2*w_PE', method = 'euler'))
    PE2L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
for k4 in range(0,6):
    PE1L_syn.append(Synapses(PEN_groups[k4+1],EPG_groups[k4+8],Ach_eqs_PE1L,on_pre='s_ach += 1*w_PE', method = 'euler'))
    PE1L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)

#PEN9-15->EPG8-15
for k4 in range(0,7):
    PE2L_syn2.append(Synapses(PEN_groups[k4+9],EPG_groups[k4+8],Ach_eqs_PE2L2,on_pre='s_ach += 2*w_PE', method = 'euler'))
    PE2L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
for k4 in range(0,6):
    PE1L_syn2.append(Synapses(PEN_groups[k4+10],EPG_groups[k4+8],Ach_eqs_PE1L2,on_pre='s_ach += 1*w_PE', method = 'euler'))
    PE1L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)


# record model state

PRM0 = PopulationRateMonitor(EPG_groups[0])
PRM1 = PopulationRateMonitor(EPG_groups[1]) 
PRM2= PopulationRateMonitor(EPG_groups[2]) 
PRM3 = PopulationRateMonitor(EPG_groups[3]) 
PRM4 = PopulationRateMonitor(EPG_groups[4]) 
PRM5 = PopulationRateMonitor(EPG_groups[5])
PRM6 = PopulationRateMonitor(EPG_groups[6]) 
PRM7= PopulationRateMonitor(EPG_groups[7]) 
PRM8 = PopulationRateMonitor(EPG_groups[8])
PRM9 = PopulationRateMonitor(EPG_groups[9])
PRM10 = PopulationRateMonitor(EPG_groups[10])
PRM11 = PopulationRateMonitor(EPG_groups[11])
PRM12 = PopulationRateMonitor(EPG_groups[12])
PRM13 = PopulationRateMonitor(EPG_groups[13])
PRM14 = PopulationRateMonitor(EPG_groups[14])
PRM15 = PopulationRateMonitor(EPG_groups[15])

PRM_pen3 = PopulationRateMonitor(PEN_groups[0])
PRM_pen1 = PopulationRateMonitor(PEN_groups[1])
PRM_pen2 = PopulationRateMonitor(PEN_groups[2])
PRM_pen3 = PopulationRateMonitor(PEN_groups[3])
PRM_pen4 = PopulationRateMonitor(PEN_groups[4])
PRM_pen5 = PopulationRateMonitor(PEN_groups[5])
PRM_pen6 = PopulationRateMonitor(PEN_groups[6])
PRM_pen7 = PopulationRateMonitor(PEN_groups[7])
PRM_pen8 = PopulationRateMonitor(PEN_groups[8])
PRM_pen9 = PopulationRateMonitor(PEN_groups[9])
PRM_pen10 = PopulationRateMonitor(PEN_groups[10])
PRM_pen11 = PopulationRateMonitor(PEN_groups[11])
PRM_pen12 = PopulationRateMonitor(PEN_groups[12])
PRM_pen13 = PopulationRateMonitor(PEN_groups[13])
PRM_pen14 = PopulationRateMonitor(PEN_groups[14])
PRM_pen15 = PopulationRateMonitor(PEN_groups[15])

SM = SpikeMonitor(EPG)
SM2 = SpikeMonitor(R)
SM3 = SpikeMonitor(PEN) 

net=Network(collect())
net.add(EPG_groups,EPG_syn,PEN_groups,PEN_syn,EP_syn,PE2R_syn,PE2L_syn,PE1R_syn,PE1L_syn,PE2R_syn2,PE2L_syn2,PE1R_syn,PE1L_syn2,PE7_syn,PE8_syn)
#run simulation
net.store()



print("Hi")
end  = time.time()
print(end - start)

######Simulation######

net.restore()
EPG_groups[3].I = 0.03
EPG_groups[4].I = 0.03
EPG_groups[5].I = 0.03
EPG_groups[6].I = 0.03
net.run(200*ms)
EPG_groups[3].I = 0.0
EPG_groups[4].I = 0.0
EPG_groups[5].I = 0.0
EPG_groups[6].I = 0.0
net.run(1*second)
last = len(SM.t)


plot(SM.t,SM.i,'.k')
plt.ylim(0,54)
plt.savefig('stable.png')

end  = time.time()
print(end - start)