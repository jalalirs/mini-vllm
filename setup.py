#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""
Setup script for mini-vLLM.

CUDA kernels are compiled separately via CMake, not by this script.
Run: cmake .. && make
Then: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="mini-vllm",
    version="0.1.0",
    description="Minimal LLM inference engine for GPT-OSS on H100",
    author="GPT-OSS Team",
    packages=find_packages(),
    package_data={
        "mini_vllm": ["_C*.so", "*.so"],  # Include compiled CUDA library
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "safetensors>=0.4.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "mini-vllm=mini_vllm.entrypoints.api_server:main",
        ],
    },
)
