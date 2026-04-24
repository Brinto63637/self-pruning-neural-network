"""
Self-Pruning Neural Network for CIFAR-10 Classification
Tredence Analytics – AI Engineering Intern Case Study

Architecture:
  - PrunableLinear: custom Linear layer with learnable sigmoid gates
  - SparsityLoss: L1 penalty on gate values encourages zero-gates
  - Training loop sweeps λ ∈ {1e-4, 1e-3, 5e-3} to show sparsity–accuracy trade-off
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
# Part 1 – PrunableLinear Layer
# ──────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that multiplies each weight
    element-wise by a learnable sigmoid gate.

    Forward pass:
        gates        = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w     = weight * gates
        output       = input @ pruned_w.T + bias

    Gradients flow through both `weight` and `gate_scores` because
    all operations are differentiable in PyTorch's autograd graph.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight & bias (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # Gate scores – same shape as weight; initialised near 0 so
        # sigmoid(gate_scores) ≈ 0.5 at start (neutral, neither open nor shut)
        self.gate_scores = nn.Parameter(torch.randn_like(self.weight) * 0.1)

    # ── helpers ──────────────────────────────────────────────

    def gates(self) -> torch.Tensor:
        """Return gate values in (0, 1)."""
        return torch.sigmoid(5 * self.gate_scores)

    def sparsity(self, threshold: float = 0.1) -> float:
        """Fraction of gates below `threshold` (considered pruned)."""
        g = self.gates().detach()
        return (g < threshold).float().mean().item()

    # ── forward ──────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates        = self.gates()                  # (out, in)
        pruned_w     = self.weight * gates           # element-wise
        return F.linear(x, pruned_w, self.bias)


# ──────────────────────────────────────────────────────────────
# Part 2 – Network definition
# ──────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Three-hidden-layer MLP for CIFAR-10 (input: 3×32×32 = 3072).
    All linear layers are PrunableLinear; activations are ReLU.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def prunable_layers(self):
        """Yield all PrunableLinear sub-modules."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of ALL gate values across every PrunableLinear layer.

        Why L1 encourages sparsity (see report section for full explanation):
          The L1 norm has a constant gradient (±1) almost everywhere,
          so it applies the same constant 'push toward zero' on every gate,
          no matter how small the gate already is. This contrasts with L2,
          whose gradient → 0 as values → 0, leaving small values alive.
        """
        all_gates = torch.cat([layer.gates().view(-1)
                                for layer in self.prunable_layers()])
        return all_gates.sum()           # all gates ≥ 0, so |g| = g

    def overall_sparsity(self, threshold: float = 0.1) -> float:
        """Network-wide fraction of pruned weights."""
        total = pruned = 0
        for layer in self.prunable_layers():
            g = layer.gates().detach()
            total  += g.numel()
            pruned += (g < threshold).sum().item()
        return pruned / total if total > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# Part 3 – Data loading
# ──────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────
# Part 4 – Training & evaluation helpers
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    lam: float, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss     = criterion(logits, labels)
        sparse_loss  = model.sparsity_loss()
        total_loss   = cls_loss + lam * sparse_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ──────────────────────────────────────────────────────────────
# Part 5 – Full experiment runner
# ──────────────────────────────────────────────────────────────

def run_experiment(lam: float, epochs: int, device: torch.device,
                   train_loader, test_loader) -> dict:
    """Train one model with sparsity coefficient `lam` and return metrics."""
    print(f"\n{'='*60}")
    print(f"  λ = {lam:.1e}  |  epochs = {epochs}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer,
                               criterion, lam, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            acc  = evaluate(model, test_loader, device)
            spar = model.overall_sparsity()
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"loss={loss:.4f} | acc={acc*100:.2f}% | "
                  f"sparsity={spar*100:.1f}%")

    final_acc  = evaluate(model, test_loader, device)
    final_spar = model.overall_sparsity()

    # Collect all gate values for plotting
    all_gates = np.concatenate([
        layer.gates().detach().cpu().numpy().flatten()
        for layer in model.prunable_layers()
    ])

    return {
        "lam"      : lam,
        "accuracy" : final_acc,
        "sparsity" : final_spar,
        "gates"    : all_gates,
        "model"    : model,
    }


# ──────────────────────────────────────────────────────────────
# Part 6 – Plotting
# ──────────────────────────────────────────────────────────────

def plot_gate_distribution(results: list, best_idx: int, save_path: str):
    """
    For each λ: histogram of final gate values.
    A successful run shows a large spike near 0 + a cluster away from 0.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    colours = ["#e63946", "#2a9d8f", "#e9c46a"]

    for ax, res, colour in zip(axes, results, colours):
        gates = res["gates"]
        ax.hist(gates, bins=80, color=colour, edgecolor="none", alpha=0.85)
        ax.axvline(x=0.01, color="black", linestyle="--",
                   linewidth=1.2, label="prune threshold (0.01)")
        ax.set_title(
            f"λ = {res['lam']:.1e}\n"
            f"Acc = {res['accuracy']*100:.1f}%  |  "
            f"Sparsity = {res['sparsity']*100:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Gate value", fontsize=10)
        ax.set_ylabel("Count",      fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle("Gate Value Distribution After Training\n"
                 "(spike at 0 = pruned weights)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Gate distribution plot saved → {save_path}")


def plot_tradeoff(results: list, save_path: str):
    """Scatter plot: sparsity vs accuracy for each λ."""
    fig, ax = plt.subplots(figsize=(7, 4))
    lams  = [r["lam"]      for r in results]
    accs  = [r["accuracy"] * 100 for r in results]
    spars = [r["sparsity"] * 100 for r in results]

    sc = ax.scatter(spars, accs, c=np.log10(lams),
                    cmap="plasma", s=180, zorder=3)
    for r, x, y in zip(results, spars, accs):
        ax.annotate(f"λ={r['lam']:.0e}", (x, y),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log₁₀(λ)", fontsize=10)
    ax.set_xlabel("Sparsity (%)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Sparsity–Accuracy Trade-off", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Trade-off plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EPOCHS     = 15          # increase for better accuracy (e.g. 50–80)
    BATCH_SIZE = 256
    LAMBDAS = [5e-3, 1e-2, 5e-2]   # low / medium / high sparsity pressure

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, EPOCHS, device, train_loader, test_loader)
        results.append(res)

    # ── Print summary table ──────────────────────────────────
    print("\n" + "="*55)
    print(f"  {'Lambda':<12}  {'Test Accuracy':>14}  {'Sparsity Level':>16}")
    print("="*55)
    for r in results:
        print(f"  {r['lam']:<12.1e}  "
              f"{r['accuracy']*100:>13.2f}%  "
              f"{r['sparsity']*100:>15.1f}%")
    print("="*55)

    # ── Identify best model (highest accuracy) ───────────────
    best_idx = max(range(len(results)),
                   key=lambda i: results[i]["accuracy"])
    print(f"\n  Best model: λ = {results[best_idx]['lam']:.1e}  "
          f"(acc = {results[best_idx]['accuracy']*100:.2f}%)")

    # ── Save plots ───────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    plot_gate_distribution(results, best_idx,
                           "outputs/gate_distributions.png")
    plot_tradeoff(results, "outputs/sparsity_accuracy_tradeoff.png")

    print("\nDone. Check the outputs/ directory for plots.")


if __name__ == "__main__":
    main()