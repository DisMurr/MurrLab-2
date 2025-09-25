"""
NOTE: this is not the most efficient way to use transformers. It's a simple implementation that infers
one token at a time to mimic the behavior of the Triton implementation.
"""

import os
from typing import Callable, List

# Transformers imports
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch


DEFAULT_TEMPERATURE = 0.0
TP = os.environ.get("TP", 2)

def load_model(checkpoint: str):
    """Serve the model directly with the Auto API using memory-safe defaults.

    Uses device_map="auto" with attention implementation eager and optional CPU/disk offload
    to reduce peak VRAM usage on single-GPU systems.
    """

    # Configure allocator to reduce fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Allow tuning via environment variables
    gpu_mem_cap = os.environ.get("GPU_MEM_CAP", "18GiB")
    offload_dir = os.environ.get("OFFLOAD_DIR", os.path.abspath("./offload"))
    os.makedirs(offload_dir, exist_ok=True)

    # If explicitly forced to CPU, avoid touching CUDA and load on CPU
    if os.environ.get("BACKEND_DEVICE", "").lower() == "cpu" or os.environ.get("FORCE_CPU", "0") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="cpu",
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_folder=offload_dir,
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                device_map="auto",
                trust_remote_code=True,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                offload_folder=offload_dir,
                offload_state_dict=True,
                max_memory={
                    0: gpu_mem_cap,  # GPU 0
                    "cpu": os.environ.get("CPU_MEM_CAP", "48GiB"),
                },
            )
        except Exception as e:
            # Fallback to CPU-only load to avoid GPU OOM during MXFP4 dequantize
            print(f"[transformers backend] GPU/auto load failed, falling back to CPU: {e}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # Disable CUDA completely for the fallback path
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                device_map="cpu",
                trust_remote_code=True,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                offload_state_dict=True,
                offload_folder=offload_dir,
            )

    model.eval()
    return model


def get_infer_next_token(model: PreTrainedModel):
    """
    Return a callable with the same shape as the original triton implementation:
      infer_next_token(tokens: List[int], temperature: float, new_request: bool) -> int

    Implementation detail:
      - We issue a single-token generation with using model.generate
      - generate handles sampling (temperature=0 => greedy, otherwise, sampling).
    """

    def infer_next_token(
        tokens: List[int],
        temperature: float = DEFAULT_TEMPERATURE,
        new_request: bool = False, # kept for interface compatibility; unused here
    ) -> int:
        tokens = torch.tensor([tokens], dtype=torch.int64, device=model.device)
        output = model.generate(tokens, max_new_tokens=1, do_sample=temperature != 0, temperature=temperature)
        return output[0, -1].tolist()

    return infer_next_token


def setup_model(checkpoint: str) -> Callable[[List[int], float, bool], int]:
    model = load_model(checkpoint)
    infer_next_token = get_infer_next_token(model)
    return infer_next_token
