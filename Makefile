.PHONY: reproduce benchmark rf sbi neural stress_damped sbi-smoke test

PYTHON ?= python

reproduce: benchmark rf sbi neural test

benchmark:
	PYTHONPATH=.:src $(PYTHON) benchmark/run_all.py

rf:
	$(PYTHON) run_benchmark.py --baseline rf

sbi:
	$(PYTHON) run_benchmark.py --baseline sbi_npe --sbi-num-simulations 100000

neural:
	$(PYTHON) run_benchmark.py --baseline neural_cnn
	$(PYTHON) run_benchmark.py --baseline neural_transformer

stress_damped:
	PYTHONPATH=.:src:baselines $(PYTHON) injections/damped_sinusoid.py

sbi-smoke:
	$(PYTHON) run_benchmark.py --baseline sbi_npe --sbi-num-simulations 5000 --sbi-max-epochs 3 --sbi-posterior-samples 64 --sbi-bootstrap 20 --sbi-checkpoint models/sbi_npe_v2_smoke.pt --sbi-output-tag sbi_npe_v2_smoke --force-train

test: sbi-smoke
	PYTHONPATH=.:src:tests $(PYTHON) tests/run_smoke_checks.py
