VARIANT ?= torch-stable-abi29-cu130-x86_64-linux

# Build only the local-hardware variant for fast iteration.
dev:
	nix run github:huggingface/kernels#kernel-builder -- build --variant $(VARIANT) --max-jobs 4 --cores 8

test:
	nix develop -L .#test --command python -m pytest tests -v
