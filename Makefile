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
exp1: 
	-mkdir data/exp1
	-rm data/exp1/*
	# Run
	parallel -j 4 -v \
			--joblog 'data/exp1.log' \
			--nice 19 --colsep ',' \
			'python theagamma/ing.py --file_name=data/exp1/result{1}-{2}.pkl --num_stim={1} --num_pop=25000 --p_stim=0.02 --output=False --stim_seed={2} --net_seed={2}' ::: 2500 ::: {1..20}
	# Extract 
	parallel -j 4 -v \
			--joblog 'data/exp1.log' \
			--nice 19 --colsep ',' \
			'python theagamma/extract.py data/exp1 data/exp1/result{1}-*.pkl'::: 2500
