# train_network Parameter Interaction Guide

## Resume Parameters Priority Order

When multiple resume parameters are set, they are evaluated in this priority order:

1. **`resume_last=True`** (highest priority)
   - Loads from `checkpoint_file_last.pt`
   - Contains the most recent training state
   - Best for continuing interrupted training

2. **`resume_checkpoint=True`**
   - Loads from `checkpoint_file.pt`
   - Contains the best model so far
   - Useful for fine-tuning from best checkpoint

3. **`resume_file="specific.pt"`**
   - Loads from the specified file path
   - For custom checkpoint management

4. **None set** (lowest priority)
   - Starts training from scratch

## Common Usage Patterns

### Pattern 1: Simple Robust Training (Recommended for Students)
```python
train_network(
    model, loss_func, train_loader, test_loader,
    checkpoint_file='model.pt',
    total_epochs=50,      # Train to 50 epochs total
    save_last=True,       # Save after every epoch
    resume_last=True,     # Resume from most recent
    device=device
)
```
**Result:** Automatically resumes from interruptions, trains to exactly 50 epochs total.

### Pattern 2: Traditional Training (Backward Compatible)
```python
train_network(
    model, loss_func, train_loader, test_loader,
    checkpoint_file='model.pt',
    epochs=20,            # Train for 20 more epochs
    resume_checkpoint=True,  # Resume from best model
    device=device
)
```
**Result:** Loads best checkpoint if exists, trains for 20 additional epochs.

### Pattern 3: Fine-tuning from Best Model
```python
train_network(
    model, loss_func, train_loader, test_loader,
    checkpoint_file='model.pt',
    resume_checkpoint=True,  # Start from best
    total_epochs=60,        # Train to epoch 60
    save_last=True,         # Track progress
    device=device
)
```
**Result:** Loads best model, continues training to reach 60 total epochs.

### Pattern 4: Manual Checkpoint Management
```python
train_network(
    model, loss_func, train_loader, test_loader,
    resume_file='checkpoint_epoch_30.pt',  # Specific checkpoint
    epochs=10,              # Train 10 more epochs
    checkpoint_file='new_model.pt',
    device=device
)
```
**Result:** Loads specific checkpoint, trains 10 more epochs, saves to new file.

## Parameter Combinations and Behaviors

| Parameters Set | Behavior | Use Case |
|---------------|----------|----------|
| `total_epochs=50, save_last=True, resume_last=True` | Resumes from last, trains to 50 total | **Best for students** |
| `epochs=20, resume_checkpoint=True` | Resumes from best, trains 20 more | Traditional fine-tuning |
| `total_epochs=50, resume_checkpoint=True` | Resumes from best, trains to 50 total | Fine-tune to specific epoch |
| `resume_last=True, resume_checkpoint=True` | **Warning shown**, uses resume_last | Conflicting (avoid) |
| `resume_file='file.pt', total_epochs=30` | Loads specific file, trains to 30 total | Custom recovery |
| No resume parameters, `epochs=20` | Fresh start, trains 20 epochs | New training |

## Messages You'll See

### When Starting Fresh:
```
Epoch: 0%|          | 0/50 [00:00<?, ?it/s, epoch=1, train_loss=2.305]
```

### When Resuming from Last:
```
Resuming from last checkpoint: model_last.pt
Resuming from epoch 10
Epoch: 20%|██      | 10/50 [00:00<?, ?it/s, epoch=11, train_loss=0.045]
```

### When Resuming from Best:
```
Resuming from best checkpoint: model.pt
Resuming from epoch 8
Epoch: 16%|█▌      | 8/50 [00:00<?, ?it/s, epoch=9, train_loss=0.052]
```

### When Training Complete:
```
Training already complete for 50 epochs.
Current epoch: 50. Increase total_epochs to continue training.
```

### After Training Finishes:
```
Best model saved at epoch 42 (val loss: 0.0234)
```

## Files Created

With `checkpoint_file='model.pt'` and `save_last=True`:

- **`model.pt`** - Best model checkpoint (lowest val/test loss or best metric)
- **`model_last.pt`** - Most recent checkpoint (for resuming)

## Key Differences: epochs vs total_epochs

| Parameter | Behavior | Example | Result |
|-----------|----------|---------|---------|
| `epochs=20` | Train for 20 MORE epochs | Start at 0 → end at 20<br>Resume from 10 → end at 30 | Relative |
| `total_epochs=20` | Train TO epoch 20 total | Start at 0 → end at 20<br>Resume from 10 → end at 20 | Absolute |

## Recommendations

### For Students (Simple & Robust):
```python
# Just add these 3 parameters to any training:
total_epochs=N,      # Train to epoch N
save_last=True,      # Save progress
resume_last=True     # Auto-resume
```

### For Research (Maximum Control):
```python
# Use traditional approach with specific control:
resume_file='specific_checkpoint.pt',
epochs=10,  # Exact number of additional epochs
checkpoint_file='experiment_v2.pt'
```

## Edge Cases Handled

1. **Missing checkpoint files** - Starts fresh with warning
2. **Corrupted checkpoint** - Falls back to fresh training
3. **Conflicting parameters** - Shows warning, uses priority order
4. **Already complete** - Returns existing results immediately
5. **Negative epochs to run** - Treated as 0, returns immediately