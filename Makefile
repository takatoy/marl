init:
	pip install pipenv --quiet
	pipenv --bare install

lint:
	pipenv run pylint --rcfile ./pylintrc ./**/*.py

hd-login:
	pipenv run hd login --email

hd-train:
	PYTHONPATH=./lib CUDA_VISIBLE_DEVICES='$(g)' pipenv run python '$(s)'

train:
	PYTHONPATH=./lib pipenv run python '$(s)'

lab:
	pipenv run jupyter lab

clean:
	rm -rf outputs
	rm -rf `pipenv --venv`
