from brian2 import ms, Hz

taum   = 20*ms   # time constant
Cm     = 0.1     # membrane capacitance
g_L    = 10   # leak conductance
E_e    = 0   # excitatory reversal potential
tau_e  = 5*ms    # excitatory synaptic time constant
E_l    = -0.07  # leak reversal potential (volt)
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


# model equations
eqs_EPG = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PE + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PE_2 : 1
Isyn_PE_1 : 1
Isyn_PE = Isyn_PE_2 + Isyn_PE_1:1
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

Ach_eqs_PE_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE_2_post = -s_ach*(v-E_ach):1 (summed)
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
