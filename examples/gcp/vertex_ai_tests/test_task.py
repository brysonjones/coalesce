"""Simple PyTorch test function to validate Vertex AI job execution."""


def run_pytorch_test():
    """Run a simple PyTorch test that sums two random values."""
    import torch

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Create two random tensors
    a = torch.rand(1, device=device)
    b = torch.rand(1, device=device)

    # Sum them
    result = a + b

    print(f"PyTorch test: {a.item():.4f} + {b.item():.4f} = {result.item():.4f}")

    return result.item()
