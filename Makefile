BASES = up

default: $(BASES)

# use bash so that `source` works
SHELL := /bin/bash

# Filter out known targets from the command line goals to get extra flags
EXTRA_ARGS := $(filter-out logs,$(MAKECMDGOALS))

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi


setup:
	if [ ! -d "venv" ]; then \
		python3.11 -m venv venv && \
		. venv/bin/activate && \
		pip install --index-url=https://artprod.dev.bloomberg.com/artifactory/api/pypi/bloomberg-pypi/simple -r requirements-dev.txt; \
	fi


up: 
	docker compose -f docker-compose.yml up

down:
	docker compose -f docker-compose.yml down

build:
	docker compose -f docker-compose.yaml build

test:
	source venv/bin/activate && python tests/test.py
