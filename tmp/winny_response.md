Hi Winny,

Good job reaching out - your instinct that something is off is correct! I found the issue and it's actually the same bug in both Part 1 and Part 2. The good news is it's a simple fix.

**The core issue: you need two separate `nn.Linear` layers, not one called twice.**

In your `SimpleBlock`, you define one linear layer and call it twice:

```python
self.Linear1 = nn.Linear(hidden_dim, hidden_dim)

def forward(self, x):
    x = F.relu(self.Linear1(x))
    x = F.relu(self.Linear1(x))
```

Your intuition makes sense on the surface - the dimensions *are* the same, so why not? But the key thing is that `nn.Linear` isn't just a shape - it's a specific set of learned weights and biases. When you call `self.Linear1` twice, you're applying the **exact same transformation** twice. It's like having one employee do two different jobs simultaneously rather than hiring two people. Each layer needs its own independent weights so it can learn its own transformation.

**The fix for Part 1:**

```python
class SimpleBlock(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SimpleBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # first layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # second layer (independent weights!)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
```

You can verify this is working by checking the model summary - you should see **134,090** total parameters instead of the 92,490 you're getting now. The `(recursive)` markers in your summary are PyTorch telling you the same layer is being reused.

**For Part 2, the same fix applies, and it matters even more.** You're also reusing a single `nn.BatchNorm1d` the same way. BatchNorm tracks running statistics (the mean and variance of what passes through it), so when you reuse one BatchNorm at two different points, those statistics get scrambled - it's trying to track two different distributions at once. That's why your BN model actually performs *worse* than your baseline - the shared BatchNorm is actively corrupting the statistics.

**The fix for Part 2:**

```python
class SimpleBlock_BN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SimpleBlock_BN, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x
```

One more small thing - in your Part 2 `Deep_MNIST_FC` class, you define `self.bn = nn.BatchNorm1d(64)` but never actually call it in `forward()`. Make sure to apply batch normalization to the input layer too:

```python
x = F.relu(self.bn(self.input_layer(x)))
```

Once you make these fixes, you should see Part 1 jump up to ~85-87% accuracy, and Part 2 with batch normalization should perform noticeably *better* than Part 1 (not worse), which is the whole point of the exercise.

The general rule: **every layer in your network needs its own `nn.Module` instance**, even if the shapes are identical. Same shape ≠ same weights!

Let me know if you have questions at office hours tomorrow.

Dr. B
