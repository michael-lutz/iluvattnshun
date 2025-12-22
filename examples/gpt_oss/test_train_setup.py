"""Quick test to validate training setup without full training."""

import torch
from train_shakespeare import GPTShakespeareConfig, GPTShakespeareTrainer


def test_training_setup():
    """Test that the training setup works without running full training."""
    print("="*70)
    print("Testing GPT-OSS Shakespeare Training Setup")
    print("="*70)

    # Create config with tiny settings for testing
    config = GPTShakespeareConfig(
        checkpoint_path="../../weights/gpt-oss-20b/original",
        max_context_length=64,  # Smaller context for testing
        learning_rate=1e-5,
        num_epochs=1,  # Just one pass for testing
        batch_size=1,
        eval_every_n_samples=10,  # Evaluate quickly
        log_every_n_seconds=5,
        tensorboard_logdir="logs/test_gpt_shakespeare",
        save_every_n_seconds=-1,  # Don't save during test
        freeze_base_model=False,
        add_extra_layer=False,
    )

    print("\n1. Creating trainer...")
    trainer = GPTShakespeareTrainer(config)
    print("   ✓ Trainer created")

    print("\n2. Loading model...")
    model = trainer.get_model().to(config.device)
    print("   ✓ Model loaded")

    print("\n3. Creating optimizer...")
    optimizer = trainer.get_optimizer(model)
    print("   ✓ Optimizer created")

    print("\n4. Testing data loader...")
    train_loader = trainer.get_train_dataloader()
    batch = next(train_loader)
    print(f"   ✓ Batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")

    print("\n5. Testing forward pass...")
    with torch.no_grad():
        logits = model(batch["input_ids"])
        print(f"   ✓ Logits shape: {logits.shape}")

    print("\n6. Testing loss computation...")
    loss, _ = trainer.get_loss(model, batch)
    print(f"   ✓ Loss: {loss.item():.4f}")

    print("\n" + "="*70)
    print("✓ All tests passed! Training setup is working correctly.")
    print("="*70)


if __name__ == "__main__":
    test_training_setup()
