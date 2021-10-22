SHELL=/bin/bash -O expand_aliases

test_ing:
	-rm data/test_ing.pkl
	python theagamma/ing.py --num_stim=100 --file_name=data/test_ing.pkl --output=False