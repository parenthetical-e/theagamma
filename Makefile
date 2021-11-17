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

# With gamma osc (using default parameters)
# ING
exp1: 
	-mkdir data/exp1
	-rm data/exp1/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp1.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp1/result{1}-{2}.pkl --num_pop=25000 --num_stim=2500 --p_stim=0.02 --stim_rate={1} --output=False --stim_seed={2} --net_seed={2}' ::: c ::: {1..20} 
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
			'python theagamma/sample.py data/exp4/sample{1}_{2} {2} data/exp1/result{1}-*.pkl' ::: 1 2 3 4 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240

# Data from exp2
exp5:
	-mkdir data/exp5
	-rm data/exp5/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp5.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp5/sample{1}_{2} {2} data/exp2/result{1}-*.pkl' ::: 1 2 3 4 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240


# Data from exp3
exp6:
	-mkdir data/exp6
	-rm data/exp6/*
	# Sample
	parallel -j 40 -v \
			--joblog 'data/exp6.log' \
			--nice 19 --colsep ',' \
			'python theagamma/sample.py data/exp6/sample{1}_{2} {2} data/exp3/result{1}-*.pkl' ::: 1 2 3 4 ::: 10 20 40 80 160 320 640 1280 2560 5120 10240