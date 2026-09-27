PYTHON ?= python

.PHONY: install data eda train pipeline test run serve docker-build docker-run

install:        ## Install runtime + development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

data:           ## Step 1: generate the synthetic dataset
	$(PYTHON) 1_generate_data.py

eda:            ## Step 2: feature engineering + EDA plots
	$(PYTHON) 2_eda_analysis.py

train:          ## Step 3: train, compare and save the model bundle
	$(PYTHON) 3_train_model.py

pipeline: data eda train  ## Run all three steps

test:           ## Run the test suite
	$(PYTHON) -m pytest

run:            ## Flask development server on http://localhost:5000
	$(PYTHON) app.py

serve:          ## Production server (settings in gunicorn.conf.py)
	gunicorn app:app

docker-build:
	docker build -t fooddelivery .

docker-run:
	docker run --rm -p 5000:5000 fooddelivery
