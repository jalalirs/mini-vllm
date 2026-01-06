# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for mini-vLLM."""

import time
import uuid
import asyncio
from typing import Optional, List, AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from mini_vllm.engine import LLMEngine, SamplingParams


# =============================================================================
# Request/Response Models (OpenAI compatible)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = 256
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# =============================================================================
# Server State
# =============================================================================

class ServerState:
    engine: Optional[LLMEngine] = None
    model_name: str = ""


state = ServerState()


# =============================================================================
# API Application
# =============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Mini-vLLM API",
        description="OpenAI-compatible API for GPT-OSS inference",
        version="0.1.0",
    )
    
    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "ok", "model": state.model_name}
    
    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [{
                "id": state.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mini-vllm",
            }]
        }
    
    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        """Create text completion."""
        if state.engine is None:
            raise HTTPException(500, "Engine not initialized")
        
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        
        if request.stream:
            return StreamingResponse(
                _stream_completion(prompts[0], params),
                media_type="text/event-stream",
            )
        
        # Non-streaming
        outputs = state.engine.generate(prompts, params)
        
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, output in enumerate(outputs):
            choices.append(CompletionChoice(
                index=i,
                text=output.text,
                finish_reason=output.finish_reason,
            ))
            total_prompt_tokens += output.prompt_tokens
            total_completion_tokens += output.completion_tokens
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=state.model_name,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create chat completion."""
        if state.engine is None:
            raise HTTPException(500, "Engine not initialized")
        
        # Format messages into prompt
        prompt = _format_chat_messages(request.messages)
        
        params = SamplingParams(
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        
        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(prompt, params),
                media_type="text/event-stream",
            )
        
        outputs = state.engine.generate([prompt], params)
        output = outputs[0]
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=state.model_name,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output.text),
                finish_reason=output.finish_reason,
            )],
            usage=Usage(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )
    
    return app


async def _stream_completion(prompt: str, params: SamplingParams) -> AsyncIterator[str]:
    """Stream completion tokens."""
    import json
    
    for output in state.engine.generate_stream(prompt, params):
        data = {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "text": output.text,
                "finish_reason": output.finish_reason,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"


async def _stream_chat_completion(prompt: str, params: SamplingParams) -> AsyncIterator[str]:
    """Stream chat completion tokens."""
    import json
    
    for output in state.engine.generate_stream(prompt, params):
        data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": output.text},
                "finish_reason": output.finish_reason,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"


def _format_chat_messages(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt.
    
    Uses a simple template. Real implementation should use model's chat template.
    """
    formatted = []
    for msg in messages:
        if msg.role == "system":
            formatted.append(f"System: {msg.content}")
        elif msg.role == "user":
            formatted.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            formatted.append(f"Assistant: {msg.content}")
    
    formatted.append("Assistant:")
    return "\n\n".join(formatted)


# =============================================================================
# Server Runner
# =============================================================================

def run_server(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
):
    """Run the API server.
    
    Args:
        model: Model path
        host: Host to bind
        port: Port to bind
        tensor_parallel_size: Number of GPUs
        dtype: Model dtype
        gpu_memory_utilization: GPU memory fraction for KV cache
        max_model_len: Maximum sequence length
    """
    import os
    from mini_vllm.distributed import get_tensor_parallel_rank, barrier
    
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if rank == 0:
        print("=" * 60)
        print("Mini-vLLM API Server")
        print("=" * 60)
        print(f"  Model: {model}")
        print(f"  Host: {host}:{port}")
        print(f"  Tensor Parallel: {tensor_parallel_size}")
        print(f"  Dtype: {dtype}")
        print("=" * 60)
    
    # Initialize engine (all ranks participate)
    if rank == 0:
        print("Loading model...")
    
    state.engine = LLMEngine.from_pretrained(
        model_path=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    state.model_name = model.split("/")[-1]
    
    if rank == 0:
        print(f"Model loaded. Starting server at {host}:{port}")
        
        # Create and run app (only on rank 0)
        app = create_app()
        uvicorn.run(app, host=host, port=port)
    else:
        # Worker ranks: wait for commands from rank 0
        print(f"[Rank {rank}] Worker ready, waiting for inference requests...")
        while True:
            barrier()  # Sync with rank 0
            # Worker logic would go here for distributed inference
            import time
            time.sleep(1)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mini-vLLM API Server")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    
    args = parser.parse_args()
    
    run_server(
        model=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()

