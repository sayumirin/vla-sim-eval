PYTHON ?= python
MODAL ?= modal

.PHONY: check plots smoke baseline plus

# Local, no GPU: recompute the statistics and redraw the figures from the committed episode records.
check:
	$(PYTHON) scripts/check_results.py

plots:
	$(PYTHON) scripts/make_figures.py

# Cloud (Modal, billed). Each target is the command that produced the committed results.
# Experiments 3 and 4 need small edits before a rerun; see their READMEs.
smoke:
	cd experiments/01_baseline && $(MODAL) run modal_pi05.py --mode smoke --max-seconds 1200

baseline:
	cd experiments/01_baseline && $(MODAL) run modal_pi05.py --mode goal --seed 0 --max-seconds 3600

plus:
	cd experiments/02_libero_plus && $(MODAL) run modal_plus.py --mode prepare
	cd experiments/02_libero_plus && $(MODAL) run modal_plus.py --mode smoke --max-seconds 600
	cd experiments/02_libero_plus && $(MODAL) run modal_plus.py --mode full --max-seconds 3600
