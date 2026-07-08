import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# -----------------------------------------------------------
# 1)  gather BN-output statistics
# -----------------------------------------------------------
def sample_batches(data_loader, device, num_batches=50):
    batches = []
    for i, (x, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        batches.append(x.to(device))   # move now or later – up to you
    return batches          # List[Tensor]  length = num_batches

# ---------------------------------------------
# revised collector: iterate over a batch list
# ---------------------------------------------
def collect_bn_activation_stats_from_batches(model: nn.Module,
                                             batches,           # List[Tensor]
                                             device: torch.device
                                             ):
    stats, handles = {}, []

    def _hook(name):
        def fn(_, __, out):
            m = out.mean(dim=(0, 2, 3)).cpu()
            v = out.var (dim=(0, 2, 3), unbiased=False).cpu()
            stats.setdefault(name, {'mean_list': [], 'var_list': []})
            stats[name]['mean_list'].append(m)
            stats[name]['var_list'].append(v)
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            handles.append(mod.register_forward_hook(_hook(name)))

    model.eval()
    with torch.no_grad():
        for x in batches:
            model(x.to(device, non_blocking=True))

    for h in handles: h.remove()

    # collapse to scalars
    final = {}
    for name in stats:
        mean_per_batch = torch.stack(stats[name]['mean_list'])   # (B,C)
        var_per_batch  = torch.stack(stats[name]['var_list'])
        final[name] = {
            'mean': mean_per_batch.mean().item(),
            'var' : var_per_batch.mean().item()
        }
    return final


# -----------------------------------------------------------
# 2)  compute ratios vs. the original model
# -----------------------------------------------------------
def compare_stats_ratio(original_stats: Dict[str, Dict[str, float]],
                        other_stats: Dict[str, Dict[str, Dict[str, float]]]
                        ) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    other_stats: {'pruned': stat_dict, 'bn_recal': stat_dict, ...}
    returns     : {'pruned': {layer: {'mean': ratio, 'var': ratio}}, ...}
    """
    ratio_results = {}
    for tag, stats in other_stats.items():
        layer_ratios = {}
        for layer in original_stats:
            # guard against any BN that disappeared during pruning
            if layer in stats:
                layer_ratios[layer] = {
                    'mean': stats[layer]['mean'] / (original_stats[layer]['mean'] + 1e-12),
                    'var' : stats[layer]['var']  / (original_stats[layer]['var']  + 1e-12)
                }
        ratio_results[tag] = layer_ratios
    return ratio_results

# -----------------------------------------------------------
# 3)  visualise
# -----------------------------------------------------------
def plot_stat_ratios(ratio_results: Dict[str, Dict[str, Dict[str, float]]],
                     title: str = "BN-output ratios (model / original)",
                     save_path: str = None):
    """
    Expects output of compare_stats_ratio.

    If ``save_path`` is given the figure is written there and closed; otherwise
    it is shown interactively (notebook behavior).
    """
    layers = list(next(iter(ratio_results.values())).keys())
    idxs   = np.arange(len(layers))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for tag, layer_ratios in ratio_results.items():
        mean_vals = [layer_ratios[l]['mean'] for l in layers]
        var_vals  = [layer_ratios[l]['var']  for l in layers]
        ax1.plot(idxs, mean_vals, marker='o', label=tag)
        ax2.plot(idxs, var_vals , marker='o', label=tag)

    ax1.set_ylabel("Mean ratio")
    ax2.set_ylabel("Var ratio")
    ax2.set_xlabel("BN layer index")
    ax1.set_title(title)
    ax1.axhline(1.0, ls='--', lw=1, alpha=.6, c='k'); ax2.axhline(1.0, ls='--', lw=1, alpha=.6, c='k')
    ax1.grid(True); ax2.grid(True)
    ax1.legend(); ax2.legend()
    plt.xticks(idxs, layers, rotation=45, ha='right')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_accuracy_comparison(accuracies: Dict[str, float],
                             title: str = "Accuracy: original vs pruned vs recalibrated",
                             save_path: str = None,
                             ylabel: str = "Top-1 accuracy (%)",
                             ymax: float = None,
                             value_fmt: str = "{:.2f}"):
    """
    Bar chart of a scalar metric for each model variant.

    ``accuracies`` maps a stage label (e.g. 'original', 'pruned', 'bn_recal')
    to its metric value. Defaults suit classification accuracy in percent; for
    other metrics (e.g. detection mAP in [0, 1]) pass ``ylabel``, ``ymax`` and
    ``value_fmt``. If ``save_path`` is given the figure is written there and
    closed; otherwise it is shown interactively.
    """
    labels = list(accuracies.keys())
    values = [accuracies[k] for k in labels]
    idxs   = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(idxs, values, color=['#4c72b0', '#dd8452', '#55a868'][:len(labels)])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(idxs)
    ax.set_xticklabels(labels)
    if ymax is None:
        ymax = max(100, max(values) * 1.1) if values else 100
    ax.set_ylim(0, ymax)
    ax.grid(True, axis='y', alpha=.3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, value_fmt.format(v),
                ha='center', va='bottom')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

