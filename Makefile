# Convenience targets for the darksiren-emri repo.

.PHONY: check lint type test validate-figures regen-figures regen-interactives help

PRODUCTION_DATA_DIR ?= simulations/cluster_run_production_h0p73_20260506/simulations

help:
	@echo "Available targets:"
	@echo "  check                 ruff + mypy + pytest (the full quality gate)"
	@echo "  lint                  ruff check + format"
	@echo "  type                  mypy on darksiren_emri/"
	@echo "  test                  pytest -m 'not gpu and not slow'"
	@echo "  validate-figures      cross-figure MAP consistency check"
	@echo "  regen-figures         regenerate every static figure on \$$PRODUCTION_DATA_DIR"
	@echo "  regen-interactives    regenerate every Plotly HTML figure on \$$PRODUCTION_DATA_DIR"

check: lint type test

lint:
	uv run ruff check darksiren_emri/ darksiren_emri_test/
	uv run ruff format --check darksiren_emri/ darksiren_emri_test/

type:
	uv run mypy darksiren_emri/

test:
	uv run pytest -m "not gpu and not slow"

# ---------------------------------------------------------------------------
# Phase H — cross-figure MAP consistency
# ---------------------------------------------------------------------------
# Runs the canonical-posterior loader against the production dataset and
# prints a one-line summary plus the formal pytest gate. Fails when any
# H0-exposing figure regresses from the canonical raw Σ log L_i MAP.
validate-figures:
	uv run python scripts/validate_figures.py $(PRODUCTION_DATA_DIR) --refresh
	uv run pytest darksiren_emri_test/plotting/test_canonical_map_consistency.py -v --no-cov

regen-figures:
	uv run python -m darksiren_emri $(PRODUCTION_DATA_DIR) \
		--generate_figures $(PRODUCTION_DATA_DIR)

regen-interactives:
	uv run python -m darksiren_emri $(PRODUCTION_DATA_DIR) \
		--generate_interactive $(PRODUCTION_DATA_DIR)
