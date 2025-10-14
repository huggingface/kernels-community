# Contributing <img src="https://github.com/user-attachments/assets/64a652f3-0cd3-4829-b3c1-df13f7933569" width="50" height="50" style="vertical-align:middle;"> to kernels-community

## Which kernels are accepted?

This repository contains kernels that are maintained, but not necessarily developed by Hugging Face. This mainly concerns kernels:

- That are developed by Hugging Face (such as yamoe).
- Kernels that are useful/high-impact, but where the upstream maintainer does not support kernels yet.

Kernels in this repository are automatically built and uploaded to [hf.co/kernels-community](https://hf.co/kernels-community).

For for your own kernels, we recommend you to develop them under your own GitHub organization and upload them to the Hugging Face Hub under your own namespace. Similarly to models and datasets, the kernels ecosystem is designed to empower the community to share their own kernels on the Hub. Of course, you are free to copy and alter our GitHub actions to build and upload kernels.

If you see an impactful kernel that you think we should host, please open a GitHub issue.

## Why is the kernel that I developed in this repository?

We packaged it as a Hub kernel because it is very impactful and most likely used by transformers, diffusers, or other Hugging Face projects. If you would like to maintain the Hub kernel yourself, we can transfer ownership to you. Please contact us through ours shared Slack collab channel (if available) or open a GitHub issue.

## How to add a new kernel?

Here is a small breakdown of the steps to add a new kernel:

1. Create a new directory in the `kernels-community` repository with the kernel name.
2. Add a `README.md` file to the directory, with a link to the kernel's source code, a kernel yaml tag, and some benchmarks.
3. Add a `flake.nix` file to the directory (you can check other kernels for examples).
4. Add a `build.toml` file to the directory where you specify which backend the kernel supports, which dependencies it has, and the source files.
5. Add a directory to put the kernel's source code (if it's not a triton kernel).
6. Add a `torch-ext` directory that will make the kernel accessible from Python using pytorch extension mechanism.
7. Add a `torch_binding.cpp` file to the `torch-ext` directory that registers the kernel as a Torch op (if it's not a triton kernel).
8. Add a directory with the same name as the kernel inside the `torch-ext` directory, and add a `__init__.py` file to the directory, there you should be able to access the kernel using the `._ops` namespace. For triton kernels, you can include all the source files in the `torch-ext` directory.
9. To test if the kernel builds successfully, you can use the `kernel-builder`.

For more details check [writing hub kernels](https://github.com/huggingface/kernel-builder/blob/main/docs/writing-kernels.md) and [building kernels with Nix](https://github.com/huggingface/kernel-builder/blob/main/docs/nix.md), and examples from [kernels-community](https://github.com/huggingface/kernels-community).

When you are done, you can open a PR to the `kernels-community` repository. Please make sure to title the PR with the kernel name, followed by a semicolon and a short description, for example: `example: add example kernel`, and do not include build outputs in the PR.

## How to benchmark a kernel?

#TODO: Add benchmarking instructions after https://github.com/huggingface/kernels-uvnotes is ready.

