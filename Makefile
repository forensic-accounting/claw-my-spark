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
		python3.12 -m venv venv && \
		. venv/bin/activate && \
		pip install -e ./forensics-pdf-mcp; \
	fi


up: 
	docker compose -f docker-compose.yml up

down:
	docker compose -f docker-compose.yml down

build:
	docker compose -f docker-compose.yaml build

test:
	source venv/bin/activate && python tests/test.py


# --- Drive Sync ---

DRIVE_GUARDS = guard-GOOGLE_APPLICATION_CREDENTIALS \
	guard-DRIVE_FOLDER_HOA \
	guard-DRIVE_FOLDER_CONDO1 \
	guard-DRIVE_FOLDER_CONDO2 \
	guard-DRIVE_FOLDER_CONDO3 \
	guard-DRIVE_FOLDER_CONDO4

S3_GUARDS = guard-S3_ENDPOINT \
	guard-S3_ACCESS_KEY \
	guard-S3_SECRET_KEY

# Populate local PDF cache from Drive (no S3 upload)
cache: $(DRIVE_GUARDS)
	source venv/bin/activate && python forensics-pdf-mcp/drive_sync.py --cache-only

# Incremental sync: cache + upload to S3
sync: $(DRIVE_GUARDS) $(S3_GUARDS)
	source venv/bin/activate && python forensics-pdf-mcp/drive_sync.py

# Full sync: clear state and re-process everything
sync-full: $(SYNC_GUARDS)
	source venv/bin/activate && python forensics-pdf-mcp/drive_sync.py --full

# Dry run: list what would be synced without doing it
sync-dry-run: $(SYNC_GUARDS)
	source venv/bin/activate && python forensics-pdf-mcp/drive_sync.py --dry-run
