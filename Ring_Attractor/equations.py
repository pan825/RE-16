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
