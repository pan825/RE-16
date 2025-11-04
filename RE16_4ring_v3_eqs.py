# model equations
# ============= Inhibitory Neurons =============
eqs_R = '''
dv/dt = (IsynEI + Isyn_ii + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
IsynE0I : 1
IsynE1I : 1
IsynE2I : 1
IsynE3I : 1
IsynEI = IsynE0I + IsynE1I + IsynE2I + IsynE3I:1
Isyn_ii:1   
'''
Ach_eqs_E0I = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynE0I_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E1I = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynE1I_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E2I = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynE2I_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E3I = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynE3I_post = -s_ach*(v-E_ach):1 (summed)
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

# ============= EPG =============

eqs_EPG = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PxE + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PxE_2 : 1
Isyn_PxE_1 : 1
Isyn_PxE = Isyn_PxE_2 + Isyn_PxE_1:1
'''

eqs_EPG0 = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PE0 + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PxE0_2 : 1
Isyn_PxE0_1 : 1
Isyn_PE0 = Isyn_PxE0_2 + Isyn_PxE0_1:1
'''

eqs_EPG1 = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PE1 + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PxE1_2 : 1
Isyn_PxE1_1 : 1
Isyn_PE1 = Isyn_PxE1_2 + Isyn_PxE1_1:1
'''

eqs_EPG2 = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PE2 + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PxE2_2 : 1
Isyn_PxE2_1 : 1
Isyn_PE2 = Isyn_PxE2_2 + Isyn_PxE2_1:1
'''

eqs_EPG3 = ''' 
dv/dt = ( Isyn + Isyn_i +Isyn_PE3 + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PxE3_2 : 1
Isyn_PxE3_1 : 1
Isyn_PE3 = Isyn_PxE3_2 + Isyn_PxE3_1:1
'''

# ============= PEN =============
eqs_PENx = '''
dv/dt = (Isyn_ppx + Isyn_EPx + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_ppx : 1
Isyn_E0Px : 1
Isyn_E1Px : 1
Isyn_E2Px : 1
Isyn_E3Px : 1
Isyn_EPx = Isyn_E0Px + Isyn_E1Px + Isyn_E2Px + Isyn_E3Px:1
'''

# ============= EPG -> EPG =============

Ach_eqs_E0E0 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E1E1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E2E2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E3E3 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# ============= PEN -> PEN =============
Ach_eqs_PxPx = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_ppx_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# ============= PEN -> EPG =============


Ach_eqs_E0Px = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_E0Px_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE0_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE0_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE0_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE0_2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E1Px = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_E1Px_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE1_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE1_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE1_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE1_2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E2Px = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_E2Px_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE2_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE2_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE2_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE2_2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_E3Px = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_E3Px_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE3_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE3_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE3_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE3_2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

#dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
