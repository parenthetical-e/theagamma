#========================================================================
import numpy as np
import fire

from fakespikes import neurons, rates
from theoc.lfp import create_lfps
from theoc.metrics import discrete_dist
from theoc.metrics import discrete_entropy
from theoc.metrics import discrete_mutual_information
from theoc.metrics import l2_error
from theoc.metrics import normalize
from theoc.oc import save_result
from scipy.signal import welch
from copy import deepcopy
from neurodsp.spectral import compute_spectrum

#========================================================================
from theagamma.util import *
from brian2 import *
from brian2.units import *
from brian2.units import amp

# Make brian shutup
prefs.codegen.target = 'numpy'  # use the Python fallback
prefs.logging.console_log_level = 'ERROR'
BrianLogger.suppress_hierarchy('brian2.codegen')
BrianLogger.suppress_name('method_choice')


def ping_coupling(
        num_pop=25000,
        num_stim=500,
        p_stim=0.2,
        g_ie=5,  # ie
        g_ei=1,  # ei
        tau_i=7.5,
        tau_e=1,
        file_name=None,
        stim_rate=2,
        output=True,
        stim_mode='drift',
        stim_seed=None,
        net_seed=None):
    """The PING network"""

    #========================================================================
    #init
    seed(net_seed)
    np.random.seed(net_seed)

    # This is just to make sure that any Brian objects created before
    # the function is called aren't included in the next run of the simulation.
    start_scope()
    t_simulation = 6 * second
    defaultclock.dt = 0.1 * ms
    srate = int(1 / float(defaultclock.dt))

    ######################################################################
    #Network Structure
    ######################################################################

    N = num_pop

    NE = int(N * 4. / 5.)
    NI = int(N / 5.)

    #-----------------------------

    prob_Pee = 0.02  #(RS->RS)
    prob_Pei = 0.02  #(RS->FS)

    prob_Pii = 0.02  #(FS->FS)
    prob_Pie = 0.02  #(FS->RS)

    prob_p = 0.02  #External

    ################################################################################
    #Reescaling Synaptic Weights based on Synaptic Decay
    ################################################################################

    tau_i = tau_i * ms
    tau_e = tau_e * ms

    #----------------------------
    #Reference synaptic weights
    #----------------------------

    Ge_extE_r = 0.8 * nS  #(External in RS)
    Ge_extI_r = 0.8 * nS  #(External in FS1a)

    Gee_r = 1 * nS  #(RS->RS)
    Gei_r = g_ei * nS  #(RS->FS1a)

    Gii_r = 5 * nS  #(FS1a->FS1a)
    Gie_r = g_ie * nS  #(FS1a->RS)

    #-----------------------------
    #This allows to study the effect of the time scales alone

    tauI_r = 5. * ms
    tauE_r = 5. * ms  #References time scales

    Ge_extE = Ge_extE_r * tauE_r / tau_e
    Ge_extI = Ge_extI_r * tauE_r / tau_e
    Gee = Gee_r * tauE_r / tau_e
    Gei = Gei_r * tauE_r / tau_e

    Gii = Gii_r * tauI_r / tau_i
    Gie = Gie_r * tauI_r / tau_i

    ################################################################################
    #Neuron Model
    ################################################################################

    #######Parameters#######

    V_reset = -65. * mvolt
    VT = -50. * mV
    Ei = -80. * mvolt
    Ee = 0. * mvolt
    t_ref = 5 * ms
    C = 150 * pF
    gL = 10 * nS

    tauw = 500 * ms

    Delay = 1.5 * ms

    #######Eleaky Heterogenities#######

    Eleaky_RS = np.full(NE, -65) * mV
    Eleaky_FS = np.full(NI, -65) * mV

    ########Equation#########

    eqs = """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + I)/C : volt (unless refractory)
    IsynE=ge*(Ee-v) : amp
    IsynI=gi*(Ei-v) : amp
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    dw/dt = (a*(v - EL) - w)/tauw : amp
    taum= C/gL : second
    I : amp
    a : siemens
    b : amp
    DeltaT: volt
    Vcut: volt
    EL : volt
    """

    ###### Initialize neuron group#############

    #FS
    neuronsI = NeuronGroup(NI,
                           eqs,
                           threshold='v>Vcut',
                           reset="v=V_reset; w+=b",
                           refractory=t_ref)
    neuronsI.a = 0 * nS
    neuronsI.b = 0. * pA
    neuronsI.DeltaT = 0.5 * mV
    neuronsI.Vcut = VT + 5 * neuronsI.DeltaT
    neuronsI.EL = Eleaky_FS

    #RS
    neuronsE = NeuronGroup(NE,
                           eqs,
                           threshold='v>Vcut',
                           reset="v=V_reset; w+=b",
                           refractory=t_ref)
    neuronsE.a = 4 * nS
    neuronsE.b = 20. * pA
    neuronsE.DeltaT = 2. * mV
    neuronsE.Vcut = VT + 5 * neuronsE.DeltaT
    neuronsE.EL = Eleaky_RS

    ####################################################################
    #Initial conditions
    ####################################################################

    #Random Membrane Potentials
    neuronsI.v = np.random.uniform(low=-65, high=-50, size=NI) * mV
    neuronsE.v = np.random.uniform(low=-65, high=-50, size=NE) * mV

    #Conductances
    neuronsI.gi = 0. * nS
    neuronsI.ge = 0. * nS
    neuronsE.gi = 0. * nS
    neuronsE.ge = 0. * nS

    #Adaptation Current
    neuronsI.w = 0.  #* amp
    neuronsE.w = 0.  #* amp

    ######################################################################
    #External Stimulus
    ######################################################################

    #!!!!!!!!!!!!!!!!!!!!!
    #From the OC codebase
    #!!!!!!!!!!!!!!!!!!!!!
    if stim_mode == "drift":
        #===================================================================
        #Drifting - Time Varying External Stimulus
        #===================================================================
        prob_ext = p_stim  # ExternalStimulus onlys
        priv_std = 0
        min_rate = 0.1
        frac_std = 0.01
        stim_std = frac_std * stim_rate

        #init poisson pop
        PoissonExternalStimulus = neurons.Spikes(int(num_stim),
                                                 float(t_simulation),
                                                 dt=float(defaultclock.dt),
                                                 seed=stim_seed)
        #create stim reference
        times = PoissonExternalStimulus.times
        stim_ref = rates.stim(times,
                              stim_rate,
                              stim_std,
                              seed=stim_seed,
                              min_rate=min_rate)

        #sample the ref
        ExtFreqPattern = PoissonExternalStimulus.poisson(stim_ref).sum(1)
    elif stim_mode == 'step':
        #================================================================
        #Stepping - Correlated and Time Varying External Stimulus
        #================================================================
        prob_ext = p_stim  # ExternalStimulus onlys

        #step parems
        f_min = 0
        f_max = stim_rate  # default was 1
        MinPlatoTime = 150 * (10**-3)
        MaxPlatoTime = 600 * (10**-3)
        TransitionTime = 50 * (10**-3)

        #times
        BaseTime = 1  #all in seconds (no units)
        T_simulation = t_simulation / second
        DT = defaultclock.dt / second

        #!
        _, ExtFreqPattern = IrregularFluctuationPattern(
            f_min, f_max, TransitionTime, MinPlatoTime, MaxPlatoTime, BaseTime,
            DT, T_simulation)

        stim_ref = ExtFreqPattern  # dummy
    else:
        raise ValueError(f"stim_mode not known")

    #======================
    # ...baack to Org code
    #======================
    rate_changes = TimedArray(ExtFreqPattern * Hz, dt=defaultclock.dt)
    ExternalStimulus = NeuronGroup(NE,
                                   'rates = rate_changes(t) : Hz',
                                   threshold='rand() < rates * dt')
    ratemonStimulus = PopulationRateMonitor(ExternalStimulus)
    spikemonStimulus = SpikeMonitor(ExternalStimulus, variables='t')

    # f_min = 0
    # f_max = 1.

    # MinPlatoTime = 150 * (10**-3)
    # MaxPlatoTime = 600 * (10**-3)

    # TransitionTime = 50 * (10**-3)
    # BaseTime = 1  #all in seconds (no units)

    # T_simulation = t_simulation / second
    # DT = defaultclock.dt / second

    # #-----

    # ExtFreqPattern_time, ExtFreqPattern = IrregularFluctuationPattern(
    #     f_min, f_max, TransitionTime, MinPlatoTime, MaxPlatoTime, BaseTime, DT,
    #     T_simulation)

    # rate_changes = TimedArray(ExtFreqPattern * Hz, dt=defaultclock.dt)

    # #-----

    # ExternalStimulus = NeuronGroup(NE,
    #                                'rates = rate_changes(t) : Hz',
    #                                threshold='rand() < rates * dt')

    #==========================================================================
    #Independent External Stimulus (constant)
    #==========================================================================

    ExtFreq = 2 * Hz
    N_ext = int(NE * prob_p)

    PoissonEonI = PoissonInput(neuronsI,
                               'ge',
                               N=N_ext,
                               rate=ExtFreq,
                               weight=Ge_extI)

    PoissonEonE = PoissonInput(neuronsE,
                               'ge',
                               N=N_ext,
                               rate=ExtFreq,
                               weight=Ge_extE)

    #==========================================================================
    #External Current
    #==========================================================================

    neuronsI.I = 0. * namp
    neuronsE.I = 0. * namp

    ##########################################################################################
    #Synaptic Connections
    ############################################################################################

    #===========================================
    #FS-RS Network (AI Network)
    #===========================================

    con_ee = Synapses(neuronsE, neuronsE, on_pre='ge_post += Gee', delay=Delay)
    con_ee.connect(p=prob_Pee)

    con_ii = Synapses(neuronsI, neuronsI, on_pre='gi_post += Gii', delay=Delay)
    con_ii.connect(p=prob_Pii)

    con_ie = Synapses(neuronsI, neuronsE, on_pre='gi_post += Gie', delay=Delay)
    con_ie.connect(p=prob_Pie)

    con_ei = Synapses(neuronsE, neuronsI, on_pre='ge_post += Gei', delay=Delay)
    con_ei.connect(p=prob_Pei)

    #===========================================
    #Time dependent External Input
    #===========================================

    con_ExtStN_E = Synapses(ExternalStimulus,
                            neuronsE,
                            on_pre='ge_post += Ge_extE',
                            delay=0. * ms)
    con_ExtStN_E.connect(p=prob_ext)

    con_ExtStN_I = Synapses(ExternalStimulus,
                            neuronsI,
                            on_pre='ge_post += Ge_extI',
                            delay=0. * ms)
    con_ExtStN_I.connect(p=prob_ext)

    #######################################################################
    # Simulation
    #######################################################################

    #Recording informations from the groups of neurons

    #FS
    statemonI = StateMonitor(neuronsI, ['v'], record=[0])
    spikemonI = SpikeMonitor(neuronsI, variables='t')
    ratemonI = PopulationRateMonitor(neuronsI)

    #RS
    statemonE = StateMonitor(neuronsE, ['v'], record=[0])
    spikemonE = SpikeMonitor(neuronsE, variables='t')
    ratemonE = PopulationRateMonitor(neuronsE)

    #RUN#
    run(t_simulation, report=ProgressBar(), report_period=0.1 * second)

    ##########################################################################
    # Mutual information calculations
    ##########################################################################

    #!!!!!!!!!!!!!!!!!!!!!
    #From the OC codebase
    #!!!!!!!!!!!!!!!!!!!!!

    #Scale stim
    y_ref = normalize(stim_ref)
    d_rescaled = {}
    d_rescaled["stim_ref"] = y_ref

    #Calc MI and H
    m = 8  #bin number
    d_mis = {}
    d_deltas = {}
    d_hs = {}
    d_py = {}
    d_errors = {}
    d_delta_errors = {}

    #Save ref H and dist
    d_py["stim_ref"] = discrete_dist(y_ref, m)
    d_hs["stim_ref"] = discrete_entropy(y_ref, m)

    #Estimate: p(y), H, MI
    #for stim, FS. RS and Iosc

    #stim
    y = normalize(ratemonStimulus.rate / Hz)
    d_rescaled["stim_p"] = y
    d_py["stim_p"] = discrete_dist(y, m)
    d_mis["stim_p"] = discrete_mutual_information(y_ref, y, m)
    d_hs["stim_p"] = discrete_entropy(y, m)
    d_errors["stim_p"] = l2_error(y_ref, y)

    #FS
    y = normalize(ratemonI.rate / Hz)
    d_rescaled["I"] = y
    d_py["I"] = discrete_dist(y, m)
    d_mis["I"] = discrete_mutual_information(y_ref, y, m)
    d_hs["I"] = discrete_entropy(y, m)
    d_errors["I"] = l2_error(y_ref, y)

    #RS
    y = normalize(ratemonE.rate / Hz)
    d_rescaled["E"] = y
    d_py["E"] = discrete_dist(y, m)
    d_mis["E"] = discrete_mutual_information(y_ref, y, m)
    d_hs["E"] = discrete_entropy(y, m)
    d_errors["E"] = l2_error(y_ref, y)

    #RS
    y = normalize((ratemonE.rate + ratemonI.rate) / Hz)
    d_rescaled["osc"] = y
    d_py["osc"] = discrete_dist(y, m)
    d_mis["osc"] = discrete_mutual_information(y_ref, y, m)
    d_hs["osc"] = discrete_entropy(y, m)
    d_errors["osc"] = l2_error(y_ref, y)

    # Change in MI
    for k in d_mis.keys():
        d_deltas[k] = d_mis[k] - d_mis["stim_p"]

    # Change in error
    for k in d_errors.keys():
        d_delta_errors[k] = d_errors[k] - d_errors["stim_p"]

    #######################################################################
    #Organizing Neuronal Spikes (for LFP Calculation)
    #######################################################################

    starting_time = 1000  #ms -> WARNING: Everything should be in ms
    ending_time = t_simulation / ms
    duration = ending_time - starting_time

    NeuronIDE = np.array(spikemonE.i)
    NeuronIDI = np.array(spikemonI.i)

    timeE = np.array(spikemonE.t / ms)  #time in ms
    timeI = np.array(spikemonI.t / ms)

    #Taking only a subgroups of neurons

    Nsub = 1000
    NEsub = int(Nsub * 4. / 5.)
    NIsub = int(Nsub / 5.)

    NeuronIDEsub = NeuronIDE[NeuronIDE < NEsub]
    NeuronIDIsub = NeuronIDI[
        NeuronIDI <
        NIsub] + NEsub  #---> this +NEsub is important for the code bellow

    timeEsub = timeE[NeuronIDE < NEsub]
    timeIsub = timeI[NeuronIDI < NIsub]

    #Cutting Transient Part:

    NeuronID_E = NeuronIDEsub[((timeEsub >= starting_time) &
                               (timeEsub < ending_time))]
    time_E = timeEsub[((timeEsub >= starting_time) & (timeEsub < ending_time))]

    NeuronID_I = NeuronIDIsub[((timeIsub >= starting_time) &
                               (timeIsub < ending_time))]
    time_I = timeIsub[((timeIsub >= starting_time) & (timeIsub < ending_time))]

    ##################################################################
    # Distributing cells in a 2D grid:
    ##################################################################

    xmax = 0.2  # size of the array (in mm)
    ymax = 0.2

    X, Y = np.random.rand(2, Nsub) * np.array([[xmax, ymax]]).T

    #0 to NEsub-1: RS
    #NEsub to NEsub+NIsub-1: FS1

    ####################################################################################################
    #LFP Parameters
    ####################################################################################################

    # Table of respective amplitudes:
    # Layer   amp_i    amp_e
    # deep    -2       -1.6
    # soma    30       4.8
    # sup     -12      2.4
    # surf    3        -0.8
    #

    # #----------------------------------------------
    # """
    # These parameters were taken from the article:
    # Telenczuk B, Telenczuk M, Destexhe A (2020)
    # A kernel-based method to calculate local field
    # potentials from networks of spiking neurons
    # Journal of Neuroscience Methods
    # """
    # #----------------------------------------------

    time_resolution = 0.1  # time resolution
    npts = int(duration / time_resolution)  # nb points in LFP vector

    xe = xmax / 2
    ye = ymax / 2  # coordinates of electrode

    va = 200  # axonal velocity (mm/sec)
    lambda_ = 0.2  # space constant (mm)
    dur = 100  # total duration of LFP waveform
    nlfp = int(dur / time_resolution)  # nb of LFP pts
    amp_e = 0.7  # uLFP amplitude for exc cells
    amp_i = -3.4  # uLFP amplitude for inh cells
    sig_i = 2.1  # std-dev of ihibition (in ms)
    sig_e = 1.5 * sig_i  # std-dev for excitation

    # amp_e = -0.16	# exc uLFP amplitude (deep layer)
    # amp_i = -0.2	# inh uLFP amplitude (deep layer)

    amp_e = 0.48  # exc uLFP amplitude (soma layer)
    amp_i = 3  # inh uLFP amplitude (soma layer)

    # amp_e = 0.24	# exc uLFP amplitude (superficial layer)
    # amp_i = -1.2	# inh uLFP amplitude (superficial layer)

    # amp_e = -0.08	# exc uLFP amplitude (surface)
    # amp_i = 0.3	# inh uLFP amplitude (surface)

    dist = np.sqrt((X - xe)**2 + (Y - ye)**2)  # distance to  electrode in mm
    delay = 10.4 + dist / va  # delay to peak (in ms)
    amp = np.exp(-dist / lambda_)
    amp[:NE] *= amp_e
    amp[NE:] *= amp_i

    s_e = 2 * sig_e * sig_e
    s_i = 2 * sig_i * sig_i

    Time_LFP = (np.arange(npts) * time_resolution) + starting_time  #in ms

    #######################################################################
    #LFP Calculation Functions
    # #######################################################################
    # """
    # This code was taken from the article: Telenczuk B, Telenczuk M, Destexhe A (2020)
    # A kernel-based method to calculate local field potentials from networks of spiking neurons
    # Journal of Neuroscience Methods

    # The code is originaly available at:
    # https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=266508&file=%2fdemo_kernel%2fdemo_lfp_kernel.py#tabs-2

    # """
    #=======================================

    def f_temporal_kernel(t, tau):
        #function defining temporal part of the kernel
        return np.exp(-(t**2) / tau)

    def calc_lfp(cells_time, cells_id, tau):
        #Calculate LFP from cells

        # this is a vectorised computation and as such it might be memory hungry
        # for long LFP series/large number of cells it may be more efficient to calculate it through looping

        spt = cells_time
        cid = cells_id

        kernel_contribs = amp[None, cid] * f_temporal_kernel(
            Time_LFP[:, None] - delay[None, cid] - spt[None, :], tau)
        lfp = kernel_contribs.sum(1)
        return lfp

    ########################################################################
    #LFP Calculation
    ########################################################################

    lfp_E = calc_lfp(time_E, NeuronID_E, s_e)
    lfp_I = calc_lfp(time_I, NeuronID_I, s_i)
    # lfp_Iosc = calc_lfp(time_Iosc, NeuronID_Iosc, s_i)

    LFP = lfp_E + lfp_I  #+ lfp_Iosc

    #log
    d_lfps = {}
    d_lfps["E"] = lfp_E
    d_lfps["I"] = lfp_I
    d_lfps["osc"] = LFP
    d_lfps["lfp"] = LFP

    ######################################################################
    #Oscilation Phase (By Hilbert Transform)
    ######################################################################

    zLFP = ZscoreNorm(LFP)

    FreqBand = 40  #Hz

    lowcut_main = FreqBand - 10
    highcut_main = FreqBand + 10  #Hz #Warning: if the bondaries are not well-chosen the result is bad

    Phase_time, LFPFiltered_main, LFP_thatOverLapsFileteredOne, Envelope_main, Oscillation_Phase_main = SignalPhase_byHilbert(
        zLFP, Time_LFP / 1000., lowcut_main, highcut_main,
        time_resolution *
        (10**-3))  #Phase from -pi to pi -> pick apears approx at 0

    #log
    d_lfps["lfp_gamma_times"] = Phase_time
    d_lfps["lfp_gamma"] = LFPFiltered_main
    d_lfps["lfp_gamma_hilbert"] = Envelope_main

    ############################################################################
    #Extract spectrum, and peak powers
    ############################################################################
    d_powers = {}
    d_centers = {}
    d_specs = {}
    for k in ["lfp", "lfp_gamma"]:
        freq, spectrum = compute_spectrum(
            d_lfps[k],
            fs=srate,
            method="welch",
            avg_type="mean",
            nperseg=srate * 2,
        )
        max_i = np.argmax(spectrum)
        d_powers[k] = spectrum[max_i]
        d_centers[k] = freq[max_i]
        d_specs[k] = deepcopy(spectrum)
    d_specs["freqs"] = freq

    ##########################################################################
    #Experimental output - select items
    ##########################################################################s

    d_spikes = {}
    d_spikes["E"] = (np.asarray(spikemonE.t), np.asarray(spikemonE.i))
    d_spikes["I"] = (np.asarray(spikemonI.t), np.asarray(spikemonI.i))
    d_spikes["stim_p"] = (np.asarray(spikemonStimulus.t),
                          np.asarray(spikemonStimulus.i))

    d_rates = {}
    d_rates["E"] = (np.asarray(ratemonE.t), np.asarray(ratemonE.rate))
    d_rates["I"] = (np.asarray(ratemonI.t), np.asarray(ratemonI.rate))
    d_rates["stim_p"] = (np.asarray(ratemonStimulus.t),
                         np.asarray(ratemonStimulus.rate))

    result = {
        'MI': d_mis,
        'dMI': d_deltas,
        'H': d_hs,
        'p_y': d_py,
        "l2": d_errors,
        "dl2": d_delta_errors,
        'spikes': d_spikes,
        'rates': d_rates,
        'norm_rates': d_rescaled,
        'lfp': d_lfps,
        'spectrum': d_specs,
        'power': d_powers,
        'center': d_centers,
        'srate': srate,
        'dt': float(defaultclock.dt),
        't': float(t_simulation)
    }

    if file_name is not None:
        save_result(file_name, result)
    if output:
        return result


# Create CL
if __name__ == "__main__":
    fire.Fire(ping_coupling)
