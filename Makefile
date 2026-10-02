# Chief Agent - common tasks
.PHONY: help setup install test lint format run dev migrate download backtest proto backup reset clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: install migrate proto ## Full first-time setup
	@echo "Setup complete. Run 'make run' and open http://localhost:8000"

install: ## Create the venv, install Python deps, build the dashboard
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	cd frontend && npm install && npm run build

test: ## Run the test suite
	.venv/bin/python -m pytest tests/ -q

test-verbose: ## Run the test suite verbosely
	.venv/bin/python -m pytest tests/ -v

run: ## Start the API and dashboard on :8000
	./scripts/run_server.sh

dev: ## Development mode: API reload + Vite HMR
	./scripts/dev.sh

migrate: ## Create/upgrade the database schema
	.venv/bin/python scripts/migrate.py --init

download: ## Download one year of 1-minute history for the watchlist
	.venv/bin/python scripts/download_data.py --years 1 --max-instruments 60 --build-higher-timeframes

proto: ## Compile the Upstox Market Data Feed V3 protobuf schema (optional)
	.venv/bin/python scripts/download_proto.py || true

backtest: ## Example backtest (override SYMBOLS/DATES)
	.venv/bin/python scripts/run_backtest.py \
		--start $(or $(START),2025-01-01) --end $(or $(END),2025-12-31) \
		--symbols $(or $(SYMBOLS),RELIANCE,INFY,HDFCBANK,TATAMOTORS,SBIN)

backup: ## Back up the database
	./scripts/backup.sh

frontend-build: ## Rebuild the dashboard
	cd frontend && npm run build

reset: ## Delete local runtime state (database, cache, backups) - DESTRUCTIVE
	rm -rf var/chief_agent.db* var/cache/* var/backups/*
	@echo "Local state cleared. Run 'make migrate' to recreate the schema."

clean: ## Remove build artefacts and caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache frontend/dist
