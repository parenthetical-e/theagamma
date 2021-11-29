SHELL=/bin/bash -O expand_aliases

test_ing:
	-rm data/test_ing.pkl
	python theagamma/ing.py --file_name=data/test_ing.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate=1 --stim_mode="drift" --output=False

test_extract:
	-rm data/test_ing*.csv
	python theagamma/extract.py data/test_ing data/*_ing.pkl

# EXPERIMENTAL PLAN 
# The big idea for these experiments is to confirm the predictions 
# made fo LNP oscillations in a semi-sort-of-realistic model. Because
# the model uses self-organized oscillations we can't ape all aspects,
# Or must change the analysis some.

# To match experiments in OC we need to:
# - Explore pop size (all - ING, PING, CHING)

# The pick a small and large pop sizes, and explore:

# Explore stim_rate (all)
# Explore g (as possible)
# Explore osc_power (as via LFP)

# Pop size sweep
# Explore n 2500 -> 250000

# -----------------------------------------------------------------
# 11/14/2021
# aaf2b9c
#
# First experiments. Explore stim rates, and saample to lower num_pop
# thus sidestepping stab. issues using smaller populations directly.
#
# With gamma osc (using default parameters)
# ING
exp1: 
	-mkdir data/exp1
	-rm data/exp1/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp1.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp1/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: 1 2 3 4 ::: {1..20} 
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp1.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp1/result{1} data/exp1/result{1}-*.pkl' ::: 1 2 3 4

# PING
exp2: 
	-mkdir data/exp2
	-rm data/exp2/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp2.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp2/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: 1 2 3 4 ::: {1..20} 
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp2.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp2/result{1} data/exp2/result{1}-*.pkl' ::: 1 2 3 4

# CHING
exp3: 
	-mkdir data/exp3
	-rm data/exp3/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp3.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp3/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: 1 2 3 4 ::: {1..20} 
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp3.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp3/result{1} data/exp3/result{1}-*.pkl' ::: 1 2 3 4

# Sample lower neuron numbers from exp1-3. Mimic pop_size in the `theoc`.

# Data from exp1
exp4:
	-mkdir data/exp4
	-rm data/exp4/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp4.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp4/sample{1} {2} data/exp1/result{1}-*.pkl' ::: 1 2 3 4 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240

# Data from exp2
exp5:
	-mkdir data/exp5
	-rm data/exp5/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp5.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp5/sample{1} {2} data/exp2/result{1}-*.pkl' ::: 1 2 3 4 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240


# Data from exp3
exp6:
	-mkdir data/exp6
	-rm data/exp6/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp6.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp6/sample{1} {2} data/exp3/result{1}-*.pkl' ::: 1 2 3 4 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240

# -----------------------------------------------------------------
# 11/18/2021
# ac6dbf3
#
# Experiments w/ osc coupling conductance.
#
# ING
# A first look at ING. The others will be harder to tune?
exp7: 
	-mkdir data/exp7
	-rm data/exp7/*
	# Run
	parallel -j 20 -v \
			--joblog 'data/exp7.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp7/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate=1 --g_i={1} --output=False --stim_seed={2} --net_seed={2}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: {1..20} 
	# Extract 
	parallel -j 20 -v \
			--joblog 'data/exp7.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp7/result{1} data/exp7/result{1}-*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0

# Data from exp1
exp8:
	-mkdir data/exp8
	-rm data/exp8/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp8.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp8/sample{1} {2} data/exp7/result{1}-*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 1280 10240

# ----------------------------------------------------------------------
# PING - consider g ie, ei
# -- ei
exp9: 
	-mkdir data/exp9
	-rm data/exp9/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp9.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp9/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate=1 --g_ie=5 --g_ei={1} --output=False --stim_seed={2} --net_seed={2}' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp9.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp9/result{1} data/exp9/result{1}-*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5

exp10:
	-mkdir data/exp10
	-rm data/exp10/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp10.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp10/sample{1} {2} data/exp9/result{1}-*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 1280 10240

# -- ie
exp11: 
	-mkdir data/exp11
	-rm data/exp11/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp11.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp11/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate=1 --g_ie={1} --g_ei=1.0 --output=False --stim_seed={2} --net_seed={2}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp11.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp11/result{1} data/exp11/result{1}-*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0

exp12:
	-mkdir data/exp12
	-rm data/exp12/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp12.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp12/sample{1} {2} data/exp11/result{1}-*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 1280 10240

# ----------------------------------------------------------------------
# CHING - consider g_e
exp13: 
	-mkdir data/exp13
	-rm data/exp13/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp13.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp13/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate=1 --g_e={1} --output=False --stim_seed={2} --net_seed={2}' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp13.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp13/result{1} data/exp13/result{1}-*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5
		
exp14:
	-mkdir data/exp14
	-rm data/exp14/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp14.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp14/sample{1} {2} data/exp13/result{1}-*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 160 10240



# -----------------------------------------------------------------
# 11/20/2021
# ac6dbf3
#
# Experiments w/ osc coupling conductance and a six fold change in stim_rate
# ranging from 0.5-3.0
#
# ING - consider g_i
exp15: 
	-mkdir data/exp15
	-rm data/exp15/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp15.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp15/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_i={1} --output=False --stim_seed={3} --net_seed={3}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp15.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp15/result-g{1}-s{2} data/exp15/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0

exp18:
	-mkdir data/exp18
	-rm data/exp18/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp18.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp18/sample-g{1}-s{2} {3} data/exp15/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# ----------------------------------------------------------------------
# PING - consider ie, ei
# -- ei
exp19: 
	-mkdir data/exp19
	-rm data/exp19/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp19.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp19/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_ie=5 --g_ei={1} --output=False --stim_seed={3} --net_seed={3}' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp19.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp19/result-g{1}-s{2} data/exp19/result-g{1}-s{2}*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0

exp20:
	-mkdir data/exp20
	-rm data/exp20/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp20.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp20/sample-g{1}-s{2} {3} data/exp19/result-g{1}-s{2}*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# -- ie
exp21: 
	-mkdir data/exp21
	-rm data/exp21/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp21.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp21/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_ie={1} --g_ei=1.0 --output=False --stim_seed={3} --net_seed={3}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 

exp21x: 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp21.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp21/result-g{1}-s{2} data/exp21/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0

exp22:
	-mkdir data/exp22
	-rm data/exp22/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp22.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp22/sample-g{1}-s{2} {3} data/exp21/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240 

# ----------------------------------------------------------------------
# CHING - consider g_e
exp23: 
	-mkdir data/exp23
	-rm data/exp23/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp23.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp23/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_e={1} --output=False --stim_seed={3} --net_seed={3}' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp23.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp23/result-g{1}-s{2} data/exp23/result-g{1}-s{2}*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0
		
exp24:
	-mkdir data/exp24
	-rm data/exp24/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp24.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp24/sample-g{1}-s{2} {3} data/exp23/result-g{1}-s{2}*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 160 10240


# -----------------------------------------------------------------
# 11/21/2021
# 9816a46
#
# Rerun exp15-18 and exp23-24 because I changed the way LFP was estimated.
# All models now only use E + I for the LFP spikes. This makes it consistent
# between all models. It also let's me see gamma power/MI effects in ching.
# The Ech was such a strong oscillator if it is included in the LFP is masks
# the downstream entrainment effects I am want to see.
# 
# NOTE: because PING models were already est. this way there is no need to
#       rerun them. Join exp19-22 with these recipes to get a full set of
#       data.

# ING - consider g_i
exp25: 
	-mkdir data/exp25
	-rm data/exp25/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp25.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp25/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_i={1} --output=False --stim_seed={3} --net_seed={3}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp25.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp25/result-g{1}-s{2} data/exp25/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0

# Data from exp1
exp26:
	-mkdir data/exp26
	-rm data/exp26/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp26.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp26/sample-g{1}-s{2} {3} data/exp25/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# CHING - consider g_e
exp27: 
	-mkdir data/exp27
	-rm data/exp27/*
	# Run
	-parallel -j 20 -v \
			--joblog 'data/exp27.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp27/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_e={1} --output=False --stim_seed={3} --net_seed={3}' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 20 -v \
			--joblog 'data/exp27.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp27/result-g{1}-s{2} data/exp27/result-g{1}-s{2}*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0
		
exp28:
	-mkdir data/exp28
	-rm data/exp28/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp28.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp28/sample-g{1}-s{2} {3} data/exp27/result-g{1}-s{2}*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 160 10240


# -------------------------------------------------------------------------
# 11/22/2021
# 63c3fbe
#
# Rerun experiments with higher n, to try and get the SD error bars down
# some in the paper.

# ING 
exp29: 
	-mkdir data/exp29
	-rm data/exp29/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp29.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp29/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_i={1} --output=False --stim_seed={3} --net_seed={3}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..100} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp29.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp29/result-g{1}-s{2} data/exp29/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0

exp30:
	-mkdir data/exp30
	-rm data/exp30/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp30.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp30/sample-g{1}-s{2} {3} data/exp29/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# PING 
# ei
exp31: 
	-mkdir data/exp31
	-rm data/exp31/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp31.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp31/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_ie=5 --g_ei={1} --output=False --stim_seed={3} --net_seed={3}' ::: 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..100} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp31.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp31/result-g{1}-s{2} data/exp31/result-g{1}-s{2}*.pkl' ::: 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4  ::: 0.5 1 1.5 2.0 2.5 3.0

exp32:
	-mkdir data/exp32
	-rm data/exp32/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp32.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp32/sample-g{1}-s{2} {3} data/exp31/result-g{1}-s{2}*.pkl' ::: 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# ie
exp33: 
	-mkdir data/exp33
	-rm data/exp33/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp33.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp33/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_ie={1} --g_ei=1.0 --output=False --stim_seed={3} --net_seed={3}' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..100} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp33.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp33/result-g{1}-s{2} data/exp33/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0

exp34:
	-mkdir data/exp34
	-rm data/exp34/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp34.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp34/sample-g{1}-s{2} {3} data/exp33/result-g{1}-s{2}*.pkl' ::: 3 3.5 4 4.5 5 5.5 6.0 6.5 7.0 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240 

# CHING
exp35: 
	-mkdir data/exp35
	-rm data/exp35/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp35.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp35/result-g{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --g_e={1} --output=False --stim_seed={3} --net_seed={3}' ::: 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..100} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp35.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp35/result-g{1}-s{2} data/exp35/result-g{1}-s{2}*.pkl' ::: 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4  ::: 0.5 1 1.5 2.0 2.5 3.0
		
exp36:
	-mkdir data/exp36
	-rm data/exp36/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp36.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp36/sample-g{1}-s{2} {3} data/exp35/result-g{1}-s{2}*.pkl' ::: 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 160 10240



# -----------------------------------------------------------------
# 11/29/2021
# 300c5e7
#
# Rerun pop-size with 
# - smaller stin_rate, consistent with more recent experiment
# - and to add l2_error output as a control. The latter was part of the PLOS 
# reviewer response.
#
# ING
exp38: 
	-mkdir data/exp38
	-rm data/exp38/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp38.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp38/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp38.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp38/result{1} data/exp38/result{1}-*.pkl' ::: 0.5 1 1.5 2.0 2.5 3.0

# PING
exp39: 
	-mkdir data/exp39
	-rm data/exp39/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp39.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp39/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp39.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp39/result{1} data/exp39/result{1}-*.pkl' ::: 0.5 1 1.5 2.0 2.5 3.0

# CHING
exp40: 
	-mkdir data/exp40
	-rm data/exp40/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp40.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp40/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp40.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp40/result{1} data/exp40/result{1}-*.pkl' ::: 0.5 1 1.5 2.0 2.5 3.0

# Sample lower neuron numbers from exp38-3. Mimic pop_size in the `theoc`.
# Data from exp38
exp41:
	-mkdir data/exp41
	-rm data/exp41/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp41.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp41/sample{1} {2} data/exp38/result{1}-*.pkl' ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240

# Data from exp39
exp42:
	-mkdir data/exp42
	-rm data/exp42/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp42.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp42/sample{1} {2} data/exp39/result{1}-*.pkl' ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240


# Data from exp40
exp43:
	-mkdir data/exp43
	-rm data/exp43/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp43.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp43/sample{1} {2} data/exp40/result{1}-*.pkl' ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240


# -----------------------------------------------------------------
# 11/29/2021
# 300c5e7
#
# Explore tau_e and tau_i in ING, CHING, PING. Hold g's fixed.

# ING 
exp44: 
	-mkdir data/exp44
	-rm data/exp44/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp44.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp44/result-t{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --tau_i={1} --output=False --stim_seed={3} --net_seed={3}' ::: 5 5.2 5.3 5.4 5.6 5.8 6 6.2 6.4 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp44.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp44/result-t{1}-s{2} data/exp44/result-t{1}-s{2}*.pkl' ::: 5 5.2 5.3 5.4 5.6 5.8 6 6.2 6.4 ::: 0.5 1 1.5 2.0 2.5 3.0

exp45:
	-mkdir data/exp45
	-rm data/exp45/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp45.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp45/sample-t{1}-s{2} {3} data/exp44/result-t{1}-s{2}*.pkl' ::: 5 5.2 5.3 5.4 5.6 5.8 6 6.2 6.4 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# PING 
# tau_e

exp46: 
	-mkdir data/exp46
	-rm data/exp46/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp46.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp46/result-t{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --tau_e={1} --tau_i=7.5 --output=False --stim_seed={3} --net_seed={3}' ::: 1 1.2 1.3 1.4 1.6 1.8 2 2.2 2.4  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp46.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp46/result-t{1}-s{2} data/exp46/result-t{1}-s{2}*.pkl' ::: 1 1.2 1.3 1.4 1.6 1.8 2 2.2 2.4  ::: 0.5 1 1.5 2.0 2.5 3.0

exp47:
	-mkdir data/exp47
	-rm data/exp47/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp47.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp47/sample-t{1}-s{2} {3} data/exp46/result-t{1}-s{2}*.pkl' ::: 1 1.2 1.3 1.4 1.6 1.8 2 2.2 2.4  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240

# tau_i
exp48: 
	-mkdir data/exp48
	-rm data/exp48/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp48.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ping.py --file_name=data/exp48/result-t{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --tau_e=1.0 --tau_i={1} --output=False --stim_seed={3} --net_seed={3}' ::: 6.5 6.75 7.0 7.25 7.5 7.75 8.0 8.25 8.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp48.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp48/result-t{1}-s{2} data/exp48/result-t{1}-s{2}*.pkl' ::: 6.5 6.75 7.0 7.25 7.5 7.75 8.0 8.25 8.5 ::: 0.5 1 1.5 2.0 2.5 3.0

exp49:
	-mkdir data/exp49
	-rm data/exp49/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp49.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp49/sample-t{1}-s{2} {3} data/exp48/result-t{1}-s{2}*.pkl' ::: 6.5 6.75 7.0 7.25 7.5 7.75 8.0 8.25 8.5 ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 1280 10240 

# CHING
# tau_e
exp50: 
	-mkdir data/exp50
	-rm data/exp50/*
	# Run
	-parallel -j 30 -v \
			--joblog 'data/exp50.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ching.py --file_name=data/exp50/result-t{1}-s{2}-{3}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={2} --tau_e={1} --output=False --stim_seed={3} --net_seed={3}' ::: 4.0 4.25 4.5 4.75 5 5.25 5.5 5.75 6.0  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: {1..20} 
	# Extract 
	-parallel -j 30 -v \
			--joblog 'data/exp50.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp50/result-t{1}-s{2} data/exp50/result-t{1}-s{2}*.pkl' ::: 4.0 4.25 4.5 4.75 5 5.25 5.5 5.75 6.0  ::: 0.5 1 1.5 2.0 2.5 3.0
		
exp51:
	-mkdir data/exp51
	-rm data/exp51/*
	# Sample
	-parallel -j 40 -v \
			--joblog 'data/exp51.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp51/sample-t{1}-s{2} {3} data/exp50/result-t{1}-s{2}*.pkl' ::: 4.0 4.25 4.5 4.75 5 5.25 5.5 5.75 6.0  ::: 0.5 1 1.5 2.0 2.5 3.0 ::: 160 10240
