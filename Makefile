# Default target
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make setup-pixi		- Creates symbolic link of .pixi directory in $MYDATA"
	@echo "  make share_env			- Copies the default environment to the current date path and links it to the latest path"
	@echo "  make copy_env          - Copies the default environment to the current date path"
	@echo "  make link_env          - Copies the shared enviornment of the current date to the latest path"


CUR_DIR := $(shell basename $(shell pwd))
DATE := $(shell date +%Y_%m_%d)
NEW_ENV_PATH := /hpc/user_apps/royerlab/conda_envs/trackedit
LATEST_ENV_PATH := /hpc/user_apps/royerlab/conda_envs/trackedit

CONDA_CMD=conda

.PHONY: setup-pixi
setup-pixi:
	mkdir -p $(MYDATA)/$(CUR_DIR)/.pixi
	ln -s $(MYDATA)/$(CUR_DIR)/.pixi .
	@echo "Pixi setup complete"

.PHONY: copy_env
copy_env:
	@$(CONDA_CMD) create -p $(NEW_ENV_PATH) --clone .pixi/envs/default --yes
	@echo "Copied default environment to $(NEW_ENV_PATH)"

.PHONY: link_env
link_env:
	ln -s $(NEW_ENV_PATH) $(LATEST_ENV_PATH)
	@echo "Linked $(LATEST_ENV_PATH) to $(NEW_ENV_PATH)"

.PHONY: shared
share_env: copy_env link_env
	@echo "Pixi shared environment setup complete"
