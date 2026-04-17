"""
pAIge GPU Verification Script
Validates RTX 3060 setup for fine-tuning workload
"""

import sys

def check_torch():
    """Verify PyTorch installation and CUDA support"""
    try:
        import torch
        print("✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        return torch
    except ImportError as e:
        print("✗ PyTorch import failed")
        print(f"  Error: {e}")
        print("\n  Install PyTorch with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

def check_cuda(torch):
    """Verify CUDA availability"""
    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        print("  Possible causes:")
        print("  - NVIDIA drivers not installed")
        print("  - PyTorch CPU-only version installed")
        print("  - CUDA toolkit version mismatch")
        sys.exit(1)
    
    print("✓ CUDA is available")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")

def check_gpu(torch):
    """Verify GPU device details"""
    device_count = torch.cuda.device_count()
    print(f"✓ GPU device count: {device_count}")
    
    if device_count == 0:
        print("✗ No GPU devices found")
        sys.exit(1)
    
    # Check primary GPU (device 0)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)
    total_vram = gpu_props.total_memory / 1e9
    
    print(f"✓ Primary GPU: {gpu_name}")
    print(f"  Total VRAM: {total_vram:.2f} GB")
    print(f"  CUDA compute capability: {gpu_props.major}.{gpu_props.minor}")
    print(f"  Multi-processor count: {gpu_props.multi_processor_count}")
    
    # Verify RTX 3060 expectations
    if "3060" not in gpu_name:
        print(f"⚠ WARNING: Expected RTX 3060, found {gpu_name}")
    
    if total_vram < 11.5:
        print(f"⚠ WARNING: Expected ~12GB VRAM, found {total_vram:.2f}GB")
    
    return total_vram

def test_tensor_ops(torch):
    """Test basic GPU tensor operations"""
    try:
        # Create small tensor on GPU
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = torch.matmul(x, y)
        
        print("✓ GPU tensor operations working")
        
        # Check memory allocation
        allocated = torch.cuda.memory_allocated(0) / 1e6
        reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"  Memory allocated: {allocated:.2f} MB")
        print(f"  Memory reserved: {reserved:.2f} MB")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print(f"✗ GPU tensor test failed: {e}")
        return False

def check_vram_headroom(total_vram):
    """Estimate VRAM requirements for pAIge training"""
    print("\n=== VRAM Headroom Analysis ===")
    
    # Estimated requirements (conservative)
    model_base = 2.5  # Qwen3-VL-4B-Instruct base model
    lora_overhead = 0.5  # LoRA adapters
    optimizer = 2.0  # AdamW 8-bit optimizer states
    activation = 3.0  # Gradient checkpointing helps, but activations still need space
    buffer = 1.0  # Safety buffer
    
    total_required = model_base + lora_overhead + optimizer + activation + buffer
    
    print(f"Estimated VRAM usage (batch=2, grad_accum=8):")
    print(f"  Model base: {model_base:.1f} GB")
    print(f"  LoRA adapters: {lora_overhead:.1f} GB")
    print(f"  Optimizer states: {optimizer:.1f} GB")
    print(f"  Activations: {activation:.1f} GB")
    print(f"  Buffer: {buffer:.1f} GB")
    print(f"  Total estimated: {total_required:.1f} GB")
    print(f"  Available VRAM: {total_vram:.1f} GB")
    
    headroom = total_vram - total_required
    print(f"  Headroom: {headroom:.1f} GB")
    
    if headroom < 0:
        print("⚠ WARNING: May need to reduce batch size or use QLoRA 4-bit")
        print("  Fallback config: batch=1, grad_accum=16, load_in_4bit=True")
    elif headroom < 1:
        print("⚠ CAUTION: Tight fit - monitor VRAM during smoke test")
    else:
        print("✓ Sufficient headroom for planned config")

def check_optional_deps():
    """Check optional dependencies"""
    print("\n=== Optional Dependencies ===")
    
    # Unsloth
    try:
        from unsloth import FastVisionModel
        print("✓ Unsloth available (primary path)")
    except ImportError:
        print("✗ Unsloth not available (will use PEFT fallback)")
    
    # Transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
    
    # PEFT
    try:
        import peft
        print(f"✓ PEFT {peft.__version__}")
    except ImportError:
        print("✗ PEFT not installed")
    
    # BitsAndBytes
    try:
        import bitsandbytes
        print(f"✓ BitsAndBytes {bitsandbytes.__version__}")
    except ImportError:
        print("✗ BitsAndBytes not installed")

def main():
    print("=" * 60)
    print("pAIge GPU Verification - RTX 3060 Setup")
    print("=" * 60)
    print()
    
    # Core checks
    torch = check_torch()
    check_cuda(torch)
    total_vram = check_gpu(torch)
    test_tensor_ops(torch)
    
    # VRAM analysis
    check_vram_headroom(total_vram)
    
    # Optional deps
    check_optional_deps()
    
    print("\n" + "=" * 60)
    print("✓ GPU verification complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. If Unsloth is missing, install remaining deps:")
    print("   pip install -r requirements.txt")
    print("2. Run pre-flight validation checklist")
    print("3. Execute smoke test on Condition A")

if __name__ == "__main__":
    main()