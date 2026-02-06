import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'   
matplotlib.rcParams['axes.unicode_minus'] = False 
# Methods and metrics

metrics = ["r", "R²",  "MSE"]

df = pd.read_csv("data/metrics_summary.csv", index_col=0)  # Assuming the first column is method names

methods = df.index.tolist()
metricsname = ["r", "R2",  "MSE"]
means = df[[f"{m}_Mean" for m in metricsname]].to_numpy()

variances = df[[f"{m}_Var" for m in metricsname]].to_numpy()

stds = np.sqrt(variances)

simsun_path = r"C:\Windows\Fonts\simsun.ttc"
if os.path.exists(simsun_path):
    zh_font = FontProperties(fname=simsun_path, size=10)
else:
    zh_font = FontProperties(family='SimSun', size=10)
en_font = FontProperties(family='Times New Roman', size=10)
plt.rcParams['font.size'] = 10


def is_chinese(text: str) -> bool:
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def plot_grouped_bars(means, stds, methods, metrics, out_png='kfold_results.png'):
    n_methods = len(methods)
    n_metrics = len(metrics)
    # spacing settings: reduce gap between metrics slightly, increase gap between bars
    metric_gap = 0.12  # gap factor between metric groups
    intra_gap = 0.03   # extra spacing between bars within a group
    bar_width = 0.18
    x = np.arange(n_metrics) * (1.0 + metric_gap)
    # new palette: Blue, Green, Orange
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#993131FF"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax2 = ax.twinx()  # right axis for MSE
    mse_idx = None
    for j, m in enumerate(metrics):
        if m.lower() == 'mse':
            mse_idx = j

    # plot bars: metrics except MSE on left axis, MSE on right axis
    left_rects = []
    right_rects = []
    for i in range(n_methods):
        for j in range(n_metrics):
            offset = x[j] + (i - (n_methods - 1) / 2) * (bar_width + intra_gap)
            val = means[i, j]
            err = stds[i, j]
            if j == mse_idx:
                rects = ax2.bar(offset, val, width=bar_width, yerr=err, capsize=4,
                                 color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.6)
                right_rects.extend(rects)
            else:
                rects = ax.bar(offset, val, width=bar_width, yerr=err, capsize=4,
                               color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.6)
                left_rects.extend(rects)

    # x ticks (metrics) — set fonts per-label
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    for label in ax.get_xticklabels():
        txt = label.get_text()
        label.set_fontproperties(zh_font if is_chinese(txt) else en_font)

    # legend: create proxies for methods and place above to avoid
    import matplotlib.patches as mpatches
    order = [0, 3,1, 2]
    n_font = FontProperties(family='Times New Roman', size=9) 
    patches = [mpatches.Patch(color=colors[i], label=methods[i]) for i in order]
    ax.legend(handles=patches, prop=n_font, loc='upper center', 
              bbox_to_anchor=(0.5, 1.01),
              ncol=int(np.ceil(n_methods / 2)), 
              handlelength=1.0, handletextpad=0.5, columnspacing=0.4,
              labelspacing=0.2)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.grid(False)

    # Compute y-limits from data +/- std to avoid clipping error bars
    left_cols = [j for j in range(n_metrics) if j != mse_idx]
    left_means = means[:, left_cols]
    left_stds = stds[:, left_cols]
    left_min = float(np.min(left_means - left_stds))
    left_max = float(np.max(left_means + left_stds))
    left_span = left_max - left_min if left_max != left_min else max(1e-3, left_max * 0.1)
    # keep lower bound non-negative for ratio-like metrics, add small margins
    bottom_left = max(0.0, left_min - left_span * 0.1)
    top_left = left_max + left_span * 0.2
    ax.set_ylim(bottom_left, top_left)
    ax.set_ylabel('r/ R² ', fontproperties=zh_font)

    # Right axis (MSE): also use means +/- stds and ensure bottom >= 0
    mse_means = means[:, mse_idx]
    mse_stds = stds[:, mse_idx]
    mse_min = float(np.min(mse_means - mse_stds))
    mse_max = float(np.max(mse_means + mse_stds))
    mse_span = mse_max - mse_min if mse_max != mse_min else max(1e-3, mse_max * 0.1)
    bottom_mse = max(0.0, mse_min - mse_span * 0.15)
    top_mse = mse_max + mse_span * 0.2
    ax2.set_ylim(bottom_mse, top_mse)
    ax2.set_ylabel('MSE', fontproperties=zh_font)


    left_y_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    right_y_offset = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02
    for rect in left_rects:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, 
            h + left_y_offset,
            f"{h:.3f}",
            ha='center', 
            va='bottom',
            fontsize=9,
            fontproperties=en_font if not is_chinese(f"{h:.3f}") else zh_font,
            bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=1)
        )

    for rect in right_rects:
        h = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width() / 2, 
            h + right_y_offset,
            f"{h:.3f}",
            ha='center', 
            va='bottom',
            fontsize=9,
            fontproperties=en_font if not is_chinese(f"{h:.3f}") else zh_font,
            bbox=dict(facecolor='white', edgecolor='none', alpha=1, pad=1)
        )
    plt.tight_layout()

    plt.savefig(out_png, dpi=800, bbox_inches='tight')


if __name__ == '__main__':
    out_png = 'figures\\Fig9_model_comparison.png'
    plot_grouped_bars(means, stds, methods, metrics, out_png=out_png)
    plt.show()
