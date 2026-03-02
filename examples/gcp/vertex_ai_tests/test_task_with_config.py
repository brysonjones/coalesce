"""PyTorch test function that accepts a config dict."""


def run_pytorch_test_with_config(config: dict):
    """Run a PyTorch test using hyperparameters from config."""
    import torch

    # Extract config values
    learning_rate = config.get("learning_rate", 0.01)
    batch_size = config.get("batch_size", 16)
    num_iterations = config.get("num_iterations", 3)
    message = config.get("message", "No message")

    print(f"Config loaded:")
    print(f"  learning_rate: {learning_rate}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_iterations: {num_iterations}")
    print(f"  message: {message}")

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nCUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\nCUDA not available, using CPU")

    # Simulate training iterations
    print(f"\nRunning {num_iterations} iterations...")
    total = torch.zeros(1, device=device)

    for i in range(num_iterations):
        # Simulate a batch of random "losses"
        batch_loss = torch.rand(batch_size, device=device).mean()
        # Simulate gradient update
        total = total + batch_loss * learning_rate
        print(f"  Iteration {i+1}: batch_loss={batch_loss.item():.4f}, total={total.item():.6f}")

    print(f"\nFinal result: {total.item():.6f}")
    return total.item()
