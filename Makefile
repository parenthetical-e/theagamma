SHELL=/bin/bash -O expand_aliases

test_ing:
	-rm data/test_ing.pkl
	python theagamma/ing.py --num_stim=100 --file_name=data/test_ing.pkl --output=False

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
			'python theagamma/sample.py data/exp14/sample{1} {2} data/exp13/result{1}-*.pkl' ::: 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 ::: 1280 10240
