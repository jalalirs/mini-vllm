# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mini-vLLM: LoRA and SageMaker support removed


from fastapi import FastAPI


def attach_router(app: FastAPI):
    """No-op for mini-vLLM. LoRA dynamic loading is not supported."""
    pass
