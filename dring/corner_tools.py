"""Small plotting helpers for posterior samples."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde


plt.rcParams["mathtext.fontset"] = "cm"

VICE_CYAN = "#37d8ff"
VICE_PINK = "#ff5fa2"
VICE_PURPLE = "#b569ff"
SUNSET_ORG = "#ffb347"
INK_DARK = "#30303a"
LIGHT_BG = "#f7f7f9"
VICE_MONEY = "#06ffcd"
MAIN_COLOR = "#1f77b4"
REF_COLOR = "#d62728"

na_blue = LinearSegmentedColormap.from_list(
    "na_blue",
    [
        "#ebf1f7",
        "#b9c9d9",
        "#8ea6bd",
        "#2b4b68",
    ],
)


def summary_stats(samples):
    """Return an array with columns [16th, 50th, 84th] percentiles."""
    q16 = np.percentile(samples, 16, axis=0)
    q50 = np.percentile(samples, 50, axis=0)
    q84 = np.percentile(samples, 84, axis=0)
    return np.vstack([q16, q50, q84]).T


def full_range_with_pad(x, pad_frac=0.05):
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    dx = 1.0 if xmin == xmax else xmax - xmin
    return xmin - pad_frac * dx, xmax + pad_frac * dx


def default_log_flags(labels):
    return ["log" in label.lower() for label in labels]


def transform_samples(samples, labels, log_flags):
    samples_tf = np.asarray(samples, dtype=float).copy()
    labels_tf = list(labels)
    for k, do_log in enumerate(log_flags):
        if do_log:
            samples_tf[:, k] = np.log10(np.maximum(samples_tf[:, k], 1e-30))
            if "log" not in labels_tf[k].lower():
                labels_tf[k] = r"$\log_{10}($" + labels_tf[k] + "$)$"
    return samples_tf, labels_tf


def transformed_ranges(samples_tf, plot_ranges=None, log_flags=None, pad_frac=0.05):
    ranges = []
    n_dim = samples_tf.shape[1]
    for k in range(n_dim):
        if plot_ranges is not None and k in plot_ranges:
            lo, hi = plot_ranges[k]
            lo = float(lo)
            hi = float(hi)
            if log_flags is not None and log_flags[k]:
                if lo <= 0 or hi <= 0:
                    raise ValueError(f"log range for dimension {k} must be positive")
                lo, hi = np.log10(lo), np.log10(hi)
            ranges.append((lo, hi))
        else:
            ranges.append(full_range_with_pad(samples_tf[:, k], pad_frac=pad_frac))
    return ranges


def maybe_kde_contours(ax, x, y, xmin, xmax, ymin, ymax, kde_bw=0.2, panel=None):
    xy = np.vstack([x, y])
    cov = np.cov(xy)
    rank = np.linalg.matrix_rank(cov)
    cond = np.linalg.cond(cov)

    try:
        if (rank < 2) or (cond > 1e12):
            noise_level = 1e-4
            xy = xy + noise_level * np.std(xy, axis=1, keepdims=True) * np.random.randn(*xy.shape)
        kde = gaussian_kde(xy, bw_method=kde_bw)
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        z_flat = zz.ravel()
        if (not np.all(np.isfinite(z_flat))) or np.nanmax(z_flat) <= 0:
            raise ValueError("empty KDE density")
        order = np.argsort(z_flat)[::-1]
        z_sorted = z_flat[order]
        cdf = np.cumsum(z_sorted)
        if cdf[-1] <= 0:
            raise ValueError("empty KDE density")
        cdf /= cdf[-1]

        probs = [0.39, 0.86, 0.98]
        z_levels = []
        for prob in probs:
            idx = np.searchsorted(cdf, prob)
            z_levels.append(z_sorted[idx])
        z_lines = np.array(sorted(z_levels))
        z_levels = np.concatenate([z_lines, [zz.max()]])

        contour_color = "#2b4b68"
        ax.contour(xx, yy, zz, levels=[z_lines[0]], colors=contour_color, linewidths=1.5, alpha=0.1)
        ax.contour(xx, yy, zz, levels=[z_lines[1]], colors=contour_color, linewidths=1.2, alpha=0.3)
        ax.contour(xx, yy, zz, levels=[z_lines[2]], colors=contour_color, linewidths=0.8, alpha=0.8)
        dense_levels = np.linspace(z_levels[0], zz.max(), 200)
        ax.contourf(xx, yy, zz, levels=dense_levels, cmap=na_blue, alpha=0.8)
    except Exception as exc:
        if panel is None:
            print(f"[corner_like_plot] KDE failed: {exc}")
        else:
            print(f"[corner_like_plot] KDE failed on {panel} panel: {exc}")


def corner_like_plot(
    samples,
    labels=None,
    log_flags=None,
    bins=40,
    figsize=None,
    plot_ranges=None,
    color_1d="#2b4b68",
    pad_frac=0.05,
    kde_bw=0.2,
    max_scatter=200,
    labelsize=20,
    stats1=None,
    stats2=None,
):
    """Lightweight corner plot: 1D histograms plus 2D KDE/scatter panels."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_samples, n_dim)")
    n_samples, n_dim = samples.shape

    if labels is None:
        labels = [fr"$x_{{{i + 1}}}$" for i in range(n_dim)]
    if len(labels) != n_dim:
        raise ValueError("labels length must match sample dimension")
    if log_flags is None:
        log_flags = default_log_flags(labels)
    if len(log_flags) != n_dim:
        raise ValueError("log_flags length must match sample dimension")

    samples_tf, labels_tf = transform_samples(samples, labels, log_flags)
    ranges = transformed_ranges(samples_tf, plot_ranges, log_flags, pad_frac)

    if figsize is None:
        figsize = (2.2 * n_dim, 2.2 * n_dim)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize, dpi=350)
    if n_dim == 1:
        axes = np.array([[axes]])

    if n_dim == 4:
        x_locators = [
            MaxNLocator(nbins=3),
            MaxNLocator(nbins=3),
            MaxNLocator(nbins=3),
            MaxNLocator(nbins=2),
        ]
        y_locators = [
            MaxNLocator(nbins=3),
            MaxNLocator(nbins=2),
            MaxNLocator(nbins=2),
            MaxNLocator(nbins=3),
        ]
    else:
        x_locators = [MaxNLocator(nbins=4) for _ in range(n_dim)]
        y_locators = [MaxNLocator(nbins=4) for _ in range(n_dim)]

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if i < j:
                ax.set_visible(False)
                continue

            x = samples_tf[:, j]
            y = samples_tf[:, i]
            xmin, xmax = ranges[j]
            ymin, ymax = ranges[i]

            if i == j:
                ax.hist(
                    x,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.2,
                    color=color_1d,
                    alpha=0.6,
                )
                ymin_hist, ymax_hist = ax.get_ylim()
                yh1 = ymin_hist + 0.92 * (ymax_hist - ymin_hist)
                yh2 = ymin_hist + 0.14 * (ymax_hist - ymin_hist)

                if stats1 is not None:
                    stats1_arr = np.asarray(stats1, dtype=float)
                    if log_flags[j]:
                        low1, med1, high1 = np.log10(stats1_arr[j])
                    else:
                        low1, med1, high1 = stats1_arr[j]

                    ax.hlines(yh1, low1, high1, colors=MAIN_COLOR, lw=1.6)
                    ax.vlines(
                        med1,
                        yh1 - 0.02 * (ymax_hist - ymin_hist),
                        yh1 + 0.02 * (ymax_hist - ymin_hist),
                        colors=MAIN_COLOR,
                        lw=1.6,
                    )
                    if stats2 is not None:
                        stats2_arr = np.asarray(stats2, dtype=float)
                        if log_flags[j]:
                            low2, med2, high2 = np.log10(stats2_arr[j])
                        else:
                            low2, med2, high2 = stats2_arr[j]
                        ax.hlines(yh2, low2, high2, colors=REF_COLOR, lw=1.2)
                        ax.vlines(
                            med2,
                            yh2 - 0.02 * (ymax_hist - ymin_hist),
                            yh2 + 0.02 * (ymax_hist - ymin_hist),
                            colors=REF_COLOR,
                            lw=1.2,
                        )
                ax.set_xlim(xmin, xmax)
                ax.set_yticks([])
            else:
                if n_samples > max_scatter:
                    idx = np.random.choice(n_samples, size=max_scatter, replace=False)
                    xs = x[idx]
                    ys = y[idx]
                else:
                    xs = x
                    ys = y
                maybe_kde_contours(ax, x, y, xmin, xmax, ymin, ymax, kde_bw=kde_bw, panel=(i, j))
                ax.scatter(xs, ys, s=1, color="#2a3947", alpha=1, edgecolors="none", rasterized=True)
                ax.scatter(xs, ys, s=8, facecolors="none", edgecolors=VICE_PINK, linewidths=0.6, alpha=0.00008, rasterized=True)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            ax.tick_params(direction="in", top=True, right=True, width=0.6, length=3, labelsize=12)
            ax.xaxis.set_major_locator(x_locators[j])
            if i != j:
                ax.yaxis.set_major_locator(y_locators[i])
            else:
                ax.set_yticks([])

            if i == n_dim - 1:
                ax.set_xlabel(labels_tf[j], fontsize=labelsize)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != j:
                ax.set_ylabel(labels_tf[i], fontsize=labelsize)
            else:
                ax.set_yticklabels([])

    fig.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98, wspace=0.08, hspace=0.08)
    return fig, axes
