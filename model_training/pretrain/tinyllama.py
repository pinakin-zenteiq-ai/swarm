import os
import sys
import time
import math
import glob
import random
import yaml
import gc
from pathlib import Path
from typing import Optional, Tuple, Union, List
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# Memory debugging utility
def print_memory_stats(stage: str, device: int = 0):
    """Print detailed CUDA memory statistics with [MEMORY_DEBUG] marker."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        print(f"[MEMORY_DEBUG] {stage}:")
        print(f"[MEMORY_DEBUG]   Allocated: {allocated:.2f} GB")
        print(f"[MEMORY_DEBUG]   Reserved: {reserved:.2f} GB")
        print(f"[MEMORY_DEBUG]   Max Allocated: {max_allocated:.2f} GB")
        print(f"[MEMORY_DEBUG]   Free: {(reserved - allocated):.2f} GB")
        print(f"[MEMORY_DEBUG]   " + "-" * 50)

# Hugging Face Imports
from transformers import OlmoConfig, OlmoForCausalLM
from transformers.models.olmo.modeling_olmo import OlmoDecoderLayer

# Lit-GPT utilities (Keep these from your local repo)
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import step_csv_logger
from pytorch_lightning.loggers import WandbLogger

# --- Configuration & Hyperparameters ---
model_name = "olmo_100m_custom"
vocab_size = 128256  # Llama 3 style
block_size = 1024 # sequence length

# Domain Weighting Configuration
# Prefix matches the directory name; Weight is the sampling ratio
train_data_config = [
    ("mixed_fineweb_dclm_10B_llama", 0.7),
    ("stackedu_dclm_shards", 0.1),
    ("stack_edu_mixed", 0.05),
    ("fineweb-math", 0.05),
    ("fineweb-hi", 0.05),
    ("fineweb-te", 0.05),
]

# Hardware / Distributed
total_devices = 1
num_of_devices = 1
num_of_nodes = 1
global_batch_size = 128
micro_batch_size = 32  # Further reduced: 32 was using 80GB! Start with 16. 

# Optimizer / Schedule
learning_rate = 4e-4
min_lr = 1e-5
warmup_steps = 2000
max_step = math.ceil(5_000_000_000 / (global_batch_size * block_size))
weight_decay = 0.1

# Calculation of intervals
batch_size = global_batch_size // total_devices
gradient_accumulation_steps = batch_size // micro_batch_size
warmup_iters = warmup_steps * gradient_accumulation_steps
max_iters = max_step * gradient_accumulation_steps

# --- Custom Dataset Classes for .npy Files ---
class NumpyDataset(IterableDataset):
    """
    Efficient iterable dataset for tokenized .npy files.
    Each .npy file contains a 1D array of token IDs.
    """
    def __init__(
        self, 
        filenames: List[str], 
        block_size: int,
        shuffle: bool = True,
        seed: int = 42,
        num_processes: int = 1,
        process_rank: int = 0
    ):
        self.filenames = filenames
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_processes = num_processes
        self.process_rank = process_rank

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        
        # Shard files across processes and workers
        num_shards = num_workers * self.num_processes
        shard_id = self.process_rank * num_workers + worker_id
        
        # Distribute files across shards
        max_num_files = len(self.filenames) // num_shards * num_shards
        shard_filenames = self.filenames[shard_id:max_num_files:num_shards]
        
        # Create RNG for this worker
        rng = random.Random(self.seed + shard_id)
        
        if self.shuffle:
            shard_filenames = list(shard_filenames)
            rng.shuffle(shard_filenames)
        
        # Iterate through files
        for filename in shard_filenames:
            try:
                # Load the entire .npy file with memory mapping
                tokens = np.load(filename, mmap_mode='r')
                
                # Calculate how many complete sequences we can extract
                total_length = len(tokens)
                if total_length < self.block_size:
                    continue  # Skip files that are too short
                
                # Number of complete sequences (non-overlapping)
                num_sequences = total_length // self.block_size
                
                # Create indices for sequences
                indices = list(range(num_sequences))
                if self.shuffle:
                    rng.shuffle(indices)
                
                for idx in indices:
                    start_idx = idx * self.block_size
                    end_idx = start_idx + self.block_size
                    
                    # Make sure we have enough tokens
                    if end_idx >= total_length:
                        break
                    
                    # Extract sequence and convert to int64
                    # Use copy() to avoid keeping the whole mmap in memory
                    sequence = np.array(tokens[start_idx:end_idx], dtype=np.int64, copy=True)
                    yield torch.from_numpy(sequence)
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

class WeightedCombinedDataset(IterableDataset):
    """
    Combines multiple datasets with weighted sampling.
    Similar to lit_gpt's CombinedDataset but works with IterableDataset.
    """
    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: List[float],
        seed: int = 42
    ):
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

    def __iter__(self):
        # Create iterators for all datasets
        dataset_iters = [iter(dataset) for dataset in self.datasets]
        rng = random.Random(self.seed)
        
        # Track which datasets are still active
        active_datasets = list(range(len(self.datasets)))
        
        while active_datasets:
            # Sample a dataset based on weights
            active_weights = [self.weights[i] for i in active_datasets]
            total = sum(active_weights)
            active_weights = [w / total for w in active_weights]
            
            chosen_idx = rng.choices(active_datasets, weights=active_weights, k=1)[0]
            
            try:
                # Get next item from chosen dataset
                item = next(dataset_iters[chosen_idx])
                yield item
            except StopIteration:
                # This dataset is exhausted, remove it
                active_datasets.remove(chosen_idx)
                if not active_datasets:
                    break


def setup(
    data_seed: int = 3406,
    data_dir: Path = Path("/home/sashi/prasanjith/LLAMA-3.2/Hawa"), # Root containing domain folders
    out_name: str = "olmo_100m_run",
    resume: Union[bool, Path] = False,
):
    print("[MEMORY_DEBUG] ============== SETUP START ==============")
    print_memory_stats("Initial state")
    
    # FSDP Strategy for Hugging Face OLMo
    # Note: use_orig_params=True helps with activation checkpointing compatibility
    # strategy = FSDPStrategy(
    #     auto_wrap_policy={OlmoDecoderLayer}, # Wrap at the decoder layer level
    #     state_dict_type="full",
    #     sharding_strategy="FULL_SHARD",
    #     limit_all_gathers=True,
    #     # activation_checkpointing_policy={OlmoDecoderLayer},  # Enable activation checkpointing
    #     use_orig_params=True,  # Use original parameters for better compatibility
    # )
    strategy = "auto"

    fabric = L.Fabric(
        devices=num_of_devices,
        num_nodes=num_of_nodes,
        strategy=strategy,
        precision="bf16-mixed",
        loggers=[WandbLogger(name=out_name, offline=True, save_dir="./wandb_logs")]
    )
    print_memory_stats("After Fabric initialization")
    
    # Initialize Model from Config
    # To hit ~100M with 128k vocab, we must tie weights or use small hidden_size
    
    # Try Flash Attention 2, fall back to SDPA if not available
    attn_impl = "flash_attention_2"
    try:
        # Test if flash_attention_2 is available
        import flash_attn
        print("[MEMORY_DEBUG] Flash Attention 2 detected!")
    except ImportError:
        print("[MEMORY_DEBUG] Flash Attention 2 not found, using sdpa (PyTorch native)")
        attn_impl = "sdpa"  # PyTorch's scaled_dot_product_attention (memory efficient)
    
    config = OlmoConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        tie_word_embeddings=True, # Saves ~50M params (128256 * 512 * 2 = 131M -> 65M)
        max_position_embeddings=block_size,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
        use_cache=False,  # Disable KV cache for training with activation checkpointing
        attn_implementation=attn_impl,  # Use Flash Attention or SDPA to save 60+ GB!
    )
    
    print("[MEMORY_DEBUG] ============================================")
    print(f"[MEMORY_DEBUG] CRITICAL: Using {attn_impl} for memory-efficient attention")
    print("[MEMORY_DEBUG] Expected savings: ~60-70 GB on attention matrices")
    print("[MEMORY_DEBUG] ============================================")

    fabric.print(f"Instantiating OLMo with config:")
    fabric.print(f"  - vocab_size: {config.vocab_size}")
    fabric.print(f"  - hidden_size: {config.hidden_size}")
    fabric.print(f"  - num_hidden_layers: {config.num_hidden_layers}")
    fabric.print(f"  - num_attention_heads: {config.num_attention_heads}")
    fabric.print(f"  - intermediate_size: {config.intermediate_size}")
    fabric.print(f"  - tie_word_embeddings: {config.tie_word_embeddings}")
    
    # Initialize model normally (OlmoRotaryEmbedding doesn't support reset_parameters)
    with fabric.init_module(empty_init=False):
        model = OlmoForCausalLM(config)
    
    print_memory_stats("After model initialization")
    
    # Enable gradient checkpointing to save memory
    print("[MEMORY_DEBUG] Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    print("[MEMORY_DEBUG] Gradient checkpointing enabled")
    print_memory_stats("After gradient checkpointing enabled")
    
    # Calculate and print model size
    num_params = sum(p.numel() for p in model.parameters())
    fabric.print(f"  - Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"[MEMORY_DEBUG] Model param count: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"[MEMORY_DEBUG] Theoretical model size (fp32): {num_params * 4 / 1024**3:.2f} GB")
    print(f"[MEMORY_DEBUG] Theoretical model size (bf16): {num_params * 2 / 1024**3:.2f} GB")

    # Setup Dataloader (Recursive & Weighted)
    print_memory_stats("Before dataloader creation")
    train_dataloader = create_weighted_dataloader(
        fabric=fabric,
        root_dir=data_dir,
        config=train_data_config,
        block_size=block_size,
        seed=data_seed
    )
    print_memory_stats("After dataloader creation")
    
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    print_memory_stats("After fabric.setup_dataloaders")
    
    # Optimizer with weight decay (exclude biases and LayerNorms)
    # Separate parameters into decay and no_decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Don't apply weight decay to biases and LayerNorm parameters
        if 'bias' in name or 'layernorm' in name.lower() or 'layer_norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    print(f"[MEMORY_DEBUG] Decay params: {len(decay_params)}, No-decay params: {len(no_decay_params)}")
    print_memory_stats("Before optimizer creation")
    
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=(0.9, 0.95), fused=True)
    
    print_memory_stats("After optimizer creation")
    
    # Setup model and optimizer separately when using empty_init=True
    model = fabric.setup_module(model)
    print_memory_stats("After fabric.setup_module")
    
    optimizer = fabric.setup_optimizers(optimizer)
    print_memory_stats("After fabric.setup_optimizers")
    
    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}
    
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train(fabric, state, train_dataloader, out_name)


def create_weighted_dataloader(fabric, root_dir, config, block_size, seed):
    """
    Creates a weighted dataloader that combines multiple domains.
    Uses custom NumpyDataset for efficient .npy file loading.
    """
    datasets = []
    final_weights = []

    fabric.print("\nSetting up datasets:")
    for prefix, weight in config:
        domain_path = root_dir / prefix
        # Recursive glob for .npy files
        filenames = sorted(glob.glob(str(domain_path / "**" / "*.npy"), recursive=True))
        
        if not filenames:
            fabric.print(f"  ⚠️  Warning: No files for {prefix} at {domain_path}")
            continue

        fabric.print(f"  ✓ {prefix}: {len(filenames)} files, weight={weight}")
        
        dataset = NumpyDataset(
            filenames=filenames,
            block_size=block_size + 1,  # Need block_size + 1 for input and target
            shuffle=True,
            seed=seed + fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank
        )
        datasets.append(dataset)
        
        # Scaling weight by number of files
        scaled_weight = weight * len(filenames) 
        final_weights.append(scaled_weight)

    if not datasets:
        raise ValueError("No datasets found! Check your data_dir and train_data_config")

    # Normalize weights
    sum_w = sum(final_weights)
    final_weights = [w / sum_w for w in final_weights]
    
    fabric.print("\nNormalized weights:")
    for (prefix, _), norm_weight in zip(config, final_weights):
        fabric.print(f"  {prefix}: {norm_weight:.4f}")

    combined = WeightedCombinedDataset(
        datasets=datasets, 
        weights=final_weights, 
        seed=seed
    )
    
    # Use num_workers=0 to debug, or keep at 2 for production
    return DataLoader(combined, batch_size=micro_batch_size, num_workers=3, pin_memory=True)

def train(fabric, state, train_dataloader, out_name):
    """Main training loop using Hugging Face OLMo model."""
    print("[MEMORY_DEBUG] ============== TRAIN START ==============")
    print_memory_stats("Start of train()")
    
    model = state["model"]
    optimizer = state["optimizer"]
    monitor = Monitor(fabric, window_size=10)
    
    fabric.print(f"\n{'='*60}")
    fabric.print(f"Starting training: {out_name}")
    fabric.print(f"{'='*60}")
    fabric.print(f"Max iterations: {max_iters}")
    fabric.print(f"Max steps: {max_step}")
    fabric.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    fabric.print(f"Warmup iterations: {warmup_iters}")
    fabric.print(f"Global batch size: {global_batch_size}")
    fabric.print(f"Micro batch size: {micro_batch_size}")
    fabric.print(f"Learning rate: {learning_rate} -> {min_lr}")
    fabric.print(f"{'='*60}\n")
    
    total_t0 = time.perf_counter()
    print_memory_stats("Before training loop")

    for train_data in train_dataloader:
        if state["iter_num"] >= max_iters:
            fabric.print(f"\n✓ Reached max_iters={max_iters}. Training complete!")
            break
        
        # Debug first 3 iterations and every 100th iteration
        debug_this_iter = state["iter_num"] < 3 or state["iter_num"] % 100 == 0
        
        if debug_this_iter:
            print(f"[MEMORY_DEBUG] ========== ITERATION {state['iter_num']} ==========")
            print_memory_stats(f"Start of iteration {state['iter_num']}")
            
        # LR Schedule (Cosine with warmup)
        lr = get_lr(state["iter_num"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        
        # Prepare inputs: OLMo HF Model expects input_ids and labels
        input_ids = train_data[:, 0:block_size].contiguous().long()
        targets = train_data[:, 1:block_size+1].contiguous().long()
        
        if debug_this_iter:
            print(f"[MEMORY_DEBUG] Batch shape: {train_data.shape}")
            print(f"[MEMORY_DEBUG] Input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            print(f"[MEMORY_DEBUG] Targets shape: {targets.shape}, dtype: {targets.dtype}")
            print(f"[MEMORY_DEBUG] Batch memory: {train_data.element_size() * train_data.nelement() / 1024**2:.2f} MB")
            print_memory_stats("After data preparation")
        
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        
        if debug_this_iter:
            print_memory_stats("Before forward pass")
            print("[MEMORY_DEBUG] Computing forward pass...")
        
        # Forward + Backward
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            # OlmoForCausalLM automatically computes loss when labels are provided
            # Disable KV cache for training to avoid checkpoint recomputation issues
            outputs = model(input_ids, labels=targets, use_cache=False)
            loss = outputs.loss
            
            if debug_this_iter:
                print(f"[MEMORY_DEBUG] Loss: {loss.item():.4f}")
                print_memory_stats("After forward pass (before backward)")
                mem_before_bwd = torch.cuda.memory_allocated(0) / 1024**3
                print(f"[MEMORY_DEBUG] Forward pass increased memory by: {mem_before_bwd - 0.37:.2f} GB")
                print("[MEMORY_DEBUG] Expected with Flash Attention: ~3-5 GB")
                print("[MEMORY_DEBUG] If you see 60+ GB increase, Flash Attention is NOT working!")
            
            fabric.backward(loss / gradient_accumulation_steps)
            
            if debug_this_iter:
                print_memory_stats("After backward pass")

        # Optimizer step (only when not accumulating)
        if not is_accumulating:
            if debug_this_iter:
                print_memory_stats("Before optimizer step")
            
            fabric.clip_gradients(model, optimizer, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
            
            if debug_this_iter:
                print_memory_stats("After optimizer step")
                gc.collect()
                torch.cuda.empty_cache()
                print_memory_stats("After garbage collection")
            
        state["iter_num"] += 1
        
        if debug_this_iter:
            print_memory_stats(f"End of iteration {state['iter_num']}")
            print(f"[MEMORY_DEBUG] ========== END ITERATION {state['iter_num']} ==========")
        
        # Logging
        if state["iter_num"] % 10 == 0:
            t1 = time.perf_counter()
            throughput = (block_size * micro_batch_size) / (t1 - iter_t0)
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            fabric.print(
                f"iter {state['iter_num']:5d} | step {state['step_count']:5d} | "
                f"loss {loss.item():.4f} | lr {lr:.2e} | "
                f"{throughput:.0f} tok/s | {(t1-iter_t0)*1000:.1f}ms | "
                f"mem {allocated:.1f}GB"
            )

        # Checkpointing every 1000 steps
        if state["step_count"] > 0 and state["step_count"] % 1000 == 0 and not is_accumulating:
            checkpoint_path = Path(f"checkpoints/{out_name}/step-{state['step_count']:06d}.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"💾 Saving checkpoint to {checkpoint_path}")
            fabric.save(checkpoint_path, state)
    
    # Final checkpoint
    total_time = time.perf_counter() - total_t0
    fabric.print(f"\n{'='*60}")
    fabric.print(f"Training completed in {total_time/3600:.2f} hours")
    fabric.print(f"Final step: {state['step_count']}")
    fabric.print(f"{'='*60}")
    
    final_checkpoint_path = Path(f"checkpoints/{out_name}/final.pth")
    final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    fabric.save(final_checkpoint_path, state)
    fabric.print(f"💾 Saved final checkpoint to {final_checkpoint_path}")


def get_lr(it):
    """Cosine learning rate schedule with linear warmup."""
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # After max_iters, return min_lr
    if it > max_iters:
        return min_lr
    # Cosine decay between warmup and max_iters
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(setup)
