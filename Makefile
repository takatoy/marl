init:
	pip install pipenv --quiet
	pipenv --bare install

lint:
	pipenv run pylint --rcfile ./pylintrc ./**/*.py

hd-login:
	pipenv run hd login --email

hd-train:
	PYTHONPATH=./lib CUDA_VISIBLE_DEVICES=$(GPU_NUM) pipenv run hd run -n='$(HD_NAME)' python $(SCRIPT_PATH) -c='$(CONFIG_MOD)'

train:
	PYTHONPATH=./lib pipenv run python $(SCRIPT_PATH) -c='$(CONFIG_MOD)'

clean:
	rm -rf outputs
	rm -rf `pipenv --venv`
