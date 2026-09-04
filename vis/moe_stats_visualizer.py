#!/usr/bin/env python3
"""Generate decision-oriented reports from FastDeploy all-expert MoE statistics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


BG = "#F5F7FA"
PANEL = "#FFFFFF"
INK = "#18212B"
MUTED = "#65717E"
GRID = "#DCE2E8"
BLUE = "#2474B5"
TEAL = "#159A93"
RED = "#D14B4B"
AMBER = "#D99024"
PURPLE = "#8064A2"
DATASET_COLORS = (BLUE, TEAL, AMBER, PURPLE, RED)
REQUIRED_KEYS = {
    "distance_sum",
    "distance_mean",
    "expert_importance_all",
    "expert_importance_all_mean",
    "expert_importance_topk",
    "expert_importance_topk_mean",
    "route_count",
    "token_count",
    "layer_ids",
}


@dataclass(frozen=True)
class Stats:
    name: str
    path: Path
    distance_sum: np.ndarray
    distance: np.ndarray
    importance_all_sum: np.ndarray
    importance_all: np.ndarray
    importance_topk_sum: np.ndarray
    importance_topk: np.ndarray
    route_count: np.ndarray
    token_count: np.ndarray
    layer_ids: np.ndarray
    metadata: dict

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    @property
    def num_experts(self) -> int:
        return self.distance.shape[1]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": 600,
            "axes.titlepad": 10,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def scalar(value: np.ndarray) -> object:
    return value.item() if np.asarray(value).shape == () else value


def load_stats(path: Path, name: str | None = None) -> Stats:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys: {', '.join(sorted(missing))}")
        metadata_raw = scalar(data["metadata"]) if "metadata" in data.files else "{}"
        try:
            metadata = json.loads(str(metadata_raw))
        except (json.JSONDecodeError, TypeError):
            metadata = {"raw": str(metadata_raw)}
        stats = Stats(
            name=name or path.stem.removesuffix("_stats"),
            path=path,
            distance_sum=np.asarray(data["distance_sum"], dtype=np.float64),
            distance=np.asarray(data["distance_mean"], dtype=np.float64),
            importance_all_sum=np.asarray(data["expert_importance_all"], dtype=np.float64),
            importance_all=np.asarray(data["expert_importance_all_mean"], dtype=np.float64),
            importance_topk_sum=np.asarray(data["expert_importance_topk"], dtype=np.float64),
            importance_topk=np.asarray(data["expert_importance_topk_mean"], dtype=np.float64),
            route_count=np.asarray(data["route_count"], dtype=np.float64),
            token_count=np.asarray(data["token_count"], dtype=np.float64),
            layer_ids=np.asarray(data["layer_ids"], dtype=np.int64),
            metadata=metadata,
        )
    expected_2d = (stats.num_layers, stats.num_experts)
    if stats.distance.shape != (stats.num_layers, stats.num_experts, stats.num_experts):
        raise ValueError(f"{path}: invalid distance shape {stats.distance.shape}")
    if stats.distance_sum.shape != stats.distance.shape:
        raise ValueError(f"{path}: invalid distance_sum shape {stats.distance_sum.shape}")
    for label, value in (
        ("importance_all_sum", stats.importance_all_sum),
        ("importance_all", stats.importance_all),
        ("importance_topk_sum", stats.importance_topk_sum),
        ("importance_topk", stats.importance_topk),
        ("route_count", stats.route_count),
    ):
        if value.shape != expected_2d:
            raise ValueError(f"{path}: invalid {label} shape {value.shape}")
    if stats.token_count.shape != (stats.num_layers,):
        raise ValueError(f"{path}: invalid token_count shape {stats.token_count.shape}")
    if not all(np.all(np.isfinite(x)) for x in (stats.distance, stats.importance_all, stats.importance_topk)):
        raise ValueError(f"{path}: arrays contain NaN or infinity")
    return stats


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.sum(values) <= 0:
        return 0.0
    values = np.sort(np.maximum(values, 0))
    n = len(values)
    return float((2 * np.dot(np.arange(1, n + 1), values) / np.sum(values) - n - 1) / n)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=np.abs(denominator) > np.finfo(np.float64).eps,
    )


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[-1], dtype=bool)
    return matrix[..., mask]


def layer_metrics(stats: Stats) -> pd.DataFrame:
    rows = []
    for pos, layer_id in enumerate(stats.layer_ids):
        distances = off_diagonal(stats.distance[pos])
        routes = stats.route_count[pos]
        mean_route = float(np.mean(routes))
        rows.append(
            {
                "dataset": stats.name,
                "layer": int(layer_id),
                "token_count": int(stats.token_count[pos]),
                "mean_pair_distance": float(np.mean(distances)),
                "p10_pair_distance": float(np.percentile(distances, 10)),
                "min_pair_distance": float(np.min(distances)),
                "route_cv": float(np.std(routes) / mean_route) if mean_route else 0.0,
                "route_gini": gini(routes),
                "route_max_over_mean": float(np.max(routes) / mean_route) if mean_route else 0.0,
                "importance_all_total": float(np.sum(stats.importance_all[pos])),
                "importance_topk_total": float(np.sum(stats.importance_topk[pos])),
                "actual_to_potential_ratio": float(
                    safe_ratio(
                        np.asarray(np.sum(stats.importance_topk[pos])),
                        np.asarray(np.sum(stats.importance_all[pos])),
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def expert_metrics(stats: Stats, top_k: int) -> pd.DataFrame:
    records: list[dict] = []
    for pos, layer_id in enumerate(stats.layer_ids):
        distance = stats.distance[pos].copy()
        np.fill_diagonal(distance, np.inf)
        nearest = np.argmin(distance, axis=1)
        all_imp = stats.importance_all[pos]
        topk_imp = stats.importance_topk[pos]
        route_rate = safe_ratio(stats.route_count[pos], np.full(stats.num_experts, stats.token_count[pos]))
        route_share = safe_ratio(stats.route_count[pos], np.full(stats.num_experts, top_k * stats.token_count[pos]))
        gap = all_imp - topk_imp
        ratio = safe_ratio(all_imp, topk_imp)

        # A high score highlights experts with strong latent output but weak actual use.
        all_rank = pd.Series(all_imp).rank(pct=True).to_numpy()
        gap_rank = pd.Series(gap).rank(pct=True).to_numpy()
        low_route_rank = 1.0 - pd.Series(route_rate).rank(pct=True).to_numpy()
        overlooked_score = 0.45 * all_rank + 0.35 * gap_rank + 0.20 * low_route_rank
        mean_peer_distance = np.nanmean(np.where(np.isfinite(distance), distance, np.nan), axis=1)
        for expert in range(stats.num_experts):
            records.append(
                {
                    "dataset": stats.name,
                    "layer": int(layer_id),
                    "expert": expert,
                    "token_count": int(stats.token_count[pos]),
                    "route_count": int(stats.route_count[pos, expert]),
                    "route_rate": route_rate[expert],
                    "route_share": route_share[expert],
                    "importance_all_sum": stats.importance_all_sum[pos, expert],
                    "importance_all_mean": all_imp[expert],
                    "importance_topk_sum": stats.importance_topk_sum[pos, expert],
                    "importance_topk_mean": topk_imp[expert],
                    "importance_gap": gap[expert],
                    "potential_ratio": ratio[expert],
                    "capture_ratio": safe_ratio(np.asarray(topk_imp[expert]), np.asarray(all_imp[expert])).item(),
                    "mean_peer_distance": mean_peer_distance[expert],
                    "nearest_expert": int(nearest[expert]),
                    "nearest_distance": distance[expert, nearest[expert]],
                    "overlooked_score": overlooked_score[expert],
                }
            )
    return pd.DataFrame.from_records(records)


def closest_pairs(stats: Stats, limit: int) -> pd.DataFrame:
    records: list[dict] = []
    upper = np.triu_indices(stats.num_experts, k=1)
    for pos, layer_id in enumerate(stats.layer_ids):
        values = stats.distance[pos][upper]
        take = min(limit, len(values))
        indexes = np.argpartition(values, take - 1)[:take]
        indexes = indexes[np.argsort(values[indexes])]
        for rank, idx in enumerate(indexes, 1):
            records.append(
                {
                    "dataset": stats.name,
                    "layer": int(layer_id),
                    "rank": rank,
                    "expert_a": int(upper[0][idx]),
                    "expert_b": int(upper[1][idx]),
                    "distance_sum": float(stats.distance_sum[pos][upper][idx]),
                    "distance": float(values[idx]),
                }
            )
    return pd.DataFrame.from_records(records)


def add_title(fig: Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=0.975, ha="left", va="top", fontsize=19, fontweight=650, color=INK)
    fig.text(0.055, 0.94, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(True, axis=grid, alpha=0.75)
        ax.set_axisbelow(True)


def label_heatmap(ax: plt.Axes, layers: np.ndarray, experts: int) -> None:
    xstep = 16 if experts >= 64 else 8
    xticks = np.arange(0, experts, xstep)
    ax.set_xticks(xticks, [f"E{x:03d}" for x in xticks], rotation=0, fontsize=7)
    ystep = max(1, math.ceil(len(layers) / 12))
    yticks = np.arange(0, len(layers), ystep)
    ax.set_yticks(yticks, layers[yticks], fontsize=7)
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Layer")


def finite_limits(values: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    if math.isclose(float(lo), float(hi)):
        hi = lo + 1.0
    return float(lo), float(hi)


def positive_log_norm(values: np.ndarray) -> colors.Normalize:
    positive = np.asarray(values)[np.asarray(values) > 0]
    if positive.size == 0:
        return colors.Normalize(0, 1)
    vmin, vmax = np.percentile(positive, [1, 99.5])
    vmin = max(float(vmin), np.finfo(float).tiny)
    vmax = max(float(vmax), vmin * 1.01)
    return colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)


def render_heatmap(
    fig: Figure,
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    cmap: str,
    norm: colors.Normalize | None = None,
    colorbar_label: str = "",
) -> None:
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(image, ax=ax, pad=0.015, fraction=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=7, color=MUTED)


def global_overview(stats_list: Sequence[Stats], layer_frames: Sequence[pd.DataFrame]) -> Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.98, top=0.88, bottom=0.07, wspace=0.16, hspace=0.24)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    add_title(fig, "All-expert MoE statistics", "Cross-dataset overview | distance is lower when expert outputs are more similar")

    ax = axes[0]
    ax.axis("off")
    columns = ["Dataset", "Tokens", "Mean dist.", "Route CV", "Actual / potential"]
    cells = []
    for stats, frame in zip(stats_list, layer_frames):
        cells.append(
            [
                stats.name,
                f"{int(np.max(stats.token_count)):,}",
                f"{frame.mean_pair_distance.mean():.2f}",
                f"{frame.route_cv.mean():.2f}",
                f"{frame.actual_to_potential_ratio.mean():.1%}",
            ]
        )
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="left", colLoc="left", bbox=[0, 0.17, 1, 0.66])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#EAF0F5" if row == 0 else PANEL)
        if row == 0:
            cell.set_text_props(weight=600, color=INK)
    ax.text(0, 0.88, "Run summary", transform=ax.transAxes, fontsize=11, fontweight=600, color=INK)
    ax.text(0, 0.06, "Ratio = sum(actual top-k importance) / sum(all-expert potential importance)", color=MUTED, fontsize=8)

    for idx, (metric, title, ylabel) in enumerate(
        (
            ("mean_pair_distance", "Expert separation by layer", "Mean pair distance"),
            ("route_cv", "Router load imbalance", "Route count CV"),
            ("actual_to_potential_ratio", "Actual-to-potential importance ratio", "Top-k / all importance"),
        ),
        start=1,
    ):
        ax = axes[idx]
        for color, frame in zip(DATASET_COLORS, layer_frames):
            ax.plot(frame.layer, frame[metric], color=color, lw=1.8, label=frame.dataset.iloc[0])
        ax.set_title(title, loc="left")
        ax.set_xlabel("MoE layer")
        ax.set_ylabel(ylabel)
        if metric == "actual_to_potential_ratio":
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        clean_axis(ax)
        ax.legend(ncol=min(3, len(stats_list)), loc="best")
    return fig


def dataset_overview(stats: Stats, layers: pd.DataFrame) -> Figure:
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 2, left=0.05, right=0.98, top=0.90, bottom=0.055, hspace=0.25, wspace=0.13)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    add_title(
        fig,
        f"{stats.name} | layer-expert map",
        f"{stats.num_layers} MoE layers, {stats.num_experts} routed experts, {int(np.max(stats.token_count)):,} recorded tokens",
    )
    route_rate = safe_ratio(stats.route_count, stats.token_count[:, None])
    render_heatmap(fig, axes[0], route_rate, "Actual route rate", "cividis", colorbar_label="fraction of tokens")
    render_heatmap(
        fig,
        axes[1],
        stats.importance_all,
        "All-expert potential importance",
        "viridis",
        positive_log_norm(stats.importance_all),
        "mean weighted norm",
    )
    render_heatmap(
        fig,
        axes[2],
        stats.importance_topk,
        "Actual top-k importance",
        "magma",
        positive_log_norm(stats.importance_topk),
        "mean weighted norm",
    )
    gap = stats.importance_all - stats.importance_topk
    abs_limit = max(abs(x) for x in finite_limits(gap, 1, 99))
    render_heatmap(
        fig,
        axes[3],
        gap,
        "Potential minus actual importance",
        "RdBu_r",
        colors.TwoSlopeNorm(vmin=-abs_limit, vcenter=0, vmax=abs_limit),
        "positive = under-captured",
    )
    for ax in axes[:4]:
        label_heatmap(ax, stats.layer_ids, stats.num_experts)

    ax = axes[4]
    ax.plot(layers.layer, layers.mean_pair_distance, color=BLUE, lw=2, label="mean distance")
    ax.fill_between(layers.layer, layers.p10_pair_distance, layers.mean_pair_distance, color=BLUE, alpha=0.12, label="p10 to mean")
    ax.set_title("Separation trend", loc="left")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Output L2 distance")
    clean_axis(ax)
    ax.legend()

    ax = axes[5]
    ax.plot(layers.layer, layers.route_cv, color=RED, lw=2, label="route CV")
    ax.plot(layers.layer, layers.route_gini, color=AMBER, lw=1.8, label="route Gini")
    ax2 = ax.twinx()
    ax2.plot(layers.layer, layers.actual_to_potential_ratio, color=TEAL, lw=1.8, label="actual / potential")
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax2.set_ylabel("Actual / potential", color=TEAL)
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="best")
    ax.set_title("Load balance and importance ratio", loc="left")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Imbalance")
    clean_axis(ax)
    ax2.spines["top"].set_visible(False)
    return fig


def format_expert(expert: int) -> str:
    return f"E{int(expert):03d}"


def render_table(ax: plt.Axes, frame: pd.DataFrame, columns: Sequence[str], labels: Sequence[str], formats: Sequence[str], title: str) -> None:
    ax.axis("off")
    cells = []
    for _, row in frame.iterrows():
        rendered = []
        for column, fmt in zip(columns, formats):
            value = row[column]
            rendered.append(fmt.format(value))
        cells.append(rendered)
    table = ax.table(cellText=cells, colLabels=labels, cellLoc="left", colLoc="left", loc="center", bbox=[0, 0.02, 1, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#EAF0F5" if row == 0 else PANEL)
        if row == 0:
            cell.set_text_props(weight=600)
    ax.set_title(title, loc="left")


def layer_detail(stats: Stats, layer_id: int, experts: pd.DataFrame, pairs: pd.DataFrame) -> Figure:
    matches = np.where(stats.layer_ids == layer_id)[0]
    if not len(matches):
        raise ValueError(f"{stats.name}: layer {layer_id} not found; valid: {stats.layer_ids.min()}..{stats.layer_ids.max()}")
    pos = int(matches[0])
    layer_experts = experts[(experts.dataset == stats.name) & (experts.layer == layer_id)].copy()
    layer_pairs = pairs[(pairs.dataset == stats.name) & (pairs.layer == layer_id)].head(8).copy()
    fig = plt.figure(figsize=(17, 12))
    gs = fig.add_gridspec(3, 2, left=0.05, right=0.98, top=0.90, bottom=0.055, hspace=0.25, wspace=0.16)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    add_title(fig, f"{stats.name} | layer {layer_id}", "Expert similarity, importance alignment, router load, and decision candidates")

    matrix = stats.distance[pos].copy()
    non_diag = off_diagonal(matrix)
    vmin, vmax = finite_limits(non_diag, 1, 99)
    image = axes[0].imshow(matrix, interpolation="nearest", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Pairwise expert output distance", loc="left")
    ticks = np.arange(0, stats.num_experts, 16)
    axes[0].set_xticks(ticks, [format_expert(x) for x in ticks], fontsize=7)
    axes[0].set_yticks(ticks, [format_expert(x) for x in ticks], fontsize=7)
    axes[0].set_xlabel("Expert B")
    axes[0].set_ylabel("Expert A")
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)

    ax = axes[1]
    rank = layer_experts.sort_values("importance_all_mean", ascending=False).reset_index(drop=True)
    x = np.arange(len(rank))
    ax.plot(x, rank.importance_all_mean, color=TEAL, lw=1.8, label="all-expert potential")
    ax.plot(x, rank.importance_topk_mean, color=PURPLE, lw=1.5, label="actual top-k")
    ax.fill_between(x, rank.importance_topk_mean, rank.importance_all_mean, where=rank.importance_all_mean >= rank.importance_topk_mean, color=AMBER, alpha=0.18)
    ax.set_yscale("symlog", linthresh=max(np.percentile(np.abs(rank.importance_all_mean), 10), 1e-8))
    ax.set_title("Importance, sorted by potential", loc="left")
    ax.set_xlabel("Expert rank")
    ax.set_ylabel("Mean weighted output norm")
    clean_axis(ax)
    ax.legend()

    ax = axes[2]
    scatter = ax.scatter(
        layer_experts.importance_all_mean,
        layer_experts.importance_topk_mean,
        c=layer_experts.route_rate,
        cmap="viridis",
        s=34,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.35,
    )
    max_value = max(layer_experts.importance_all_mean.max(), layer_experts.importance_topk_mean.max())
    ax.plot([0, max_value], [0, max_value], color=MUTED, ls="--", lw=1, label="actual = potential")
    candidates = layer_experts.nlargest(5, "overlooked_score")
    ax.scatter(
        candidates.importance_all_mean,
        candidates.importance_topk_mean,
        s=72,
        facecolor="none",
        edgecolor=AMBER,
        linewidth=1.4,
        label="inspect candidates",
    )
    ax.set_title("Router alignment", loc="left")
    ax.set_xlabel("All-expert potential importance")
    ax.set_ylabel("Actual top-k importance")
    clean_axis(ax, "both")
    ax.legend(loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Route rate", fontsize=8)
    cbar.outline.set_visible(False)

    ax = axes[3]
    route_rank = layer_experts.sort_values("route_rate", ascending=False).reset_index(drop=True)
    bar_colors = np.where(route_rank.route_rate >= route_rank.route_rate.quantile(0.9), RED, BLUE)
    ax.bar(np.arange(len(route_rank)), route_rank.route_rate, color=bar_colors, width=0.82)
    ax.axhline(route_rank.route_rate.mean(), color=INK, lw=1, ls="--", label=f"mean {route_rank.route_rate.mean():.1%}")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title("Actual router load, descending", loc="left")
    ax.set_xlabel("Expert rank")
    ax.set_ylabel("Tokens selecting expert")
    clean_axis(ax)
    ax.legend()

    render_table(
        axes[4],
        layer_pairs,
        ("rank", "expert_a", "expert_b", "distance"),
        ("#", "Expert A", "Expert B", "Distance"),
        ("{:.0f}", "E{:03.0f}", "E{:03.0f}", "{:.4f}"),
        "Closest expert pairs | merge candidates",
    )
    candidate_table = candidates.sort_values("overlooked_score", ascending=False).head(8)
    render_table(
        axes[5],
        candidate_table,
        ("expert", "route_rate", "importance_all_mean", "importance_gap", "overlooked_score"),
        ("Expert", "Route", "Potential", "Gap", "Score"),
        ("E{:03.0f}", "{:.1%}", "{:.4g}", "{:+.4g}", "{:.3f}"),
        "High-potential, weakly captured experts | inspect candidates",
    )
    return fig


def cross_dataset_view(stats_list: Sequence[Stats], experts: pd.DataFrame) -> Figure:
    common_layers = sorted(set.intersection(*(set(x.layer_ids.tolist()) for x in stats_list)))
    common_experts = min(x.num_experts for x in stats_list)
    metrics = ("route_rate", "importance_all_mean", "importance_topk_mean")
    labels = ("Route rate", "Potential importance", "Actual top-k importance")
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.98, top=0.89, bottom=0.06, hspace=0.25, wspace=0.15)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    add_title(fig, "Cross-dataset stability", "Domain variation reveals specialist experts and dataset-sensitive routing")

    for ax, metric, label in zip(axes[:3], metrics, labels):
        pivot = experts.pivot_table(index=["layer", "expert"], columns="dataset", values=metric)
        pivot = pivot.dropna()
        mean = pivot.mean(axis=1)
        cv = safe_ratio(pivot.std(axis=1).to_numpy(), np.abs(mean.to_numpy()))
        matrix = np.full((len(common_layers), common_experts), np.nan)
        layer_index = {layer: i for i, layer in enumerate(common_layers)}
        for (layer, expert), value in zip(pivot.index, cv):
            if layer in layer_index and expert < common_experts:
                matrix[layer_index[layer], expert] = value
        _, vmax = finite_limits(matrix, 0, 98)
        render_heatmap(fig, ax, matrix, f"{label} variation", "inferno", colors.Normalize(0, vmax), "cross-dataset CV")
        label_heatmap(ax, np.asarray(common_layers), common_experts)

    ax = axes[3]
    agg = experts.groupby(["dataset", "expert"], as_index=False).agg(route_rate=("route_rate", "mean"), potential=("importance_all_mean", "mean"))
    color_map = {stats.name: DATASET_COLORS[index % len(DATASET_COLORS)] for index, stats in enumerate(stats_list)}
    for dataset in (stats.name for stats in stats_list):
        frame = agg[agg.dataset == dataset]
        color = color_map[dataset]
        ax.scatter(frame.route_rate, frame.potential, s=18, alpha=0.58, label=dataset, color=color, edgecolor="none")
    ax.set_title("Expert specialization across all layers", loc="left")
    ax.set_xlabel("Mean route rate")
    ax.set_ylabel("Mean potential importance")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_yscale("log")
    clean_axis(ax, "both")
    ax.legend()
    return fig


def resolve_inputs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.inputs:
        resolved = []
        for raw in args.inputs:
            path = Path(raw).expanduser().resolve()
            resolved.append((path.stem.removesuffix("_stats"), path))
        return resolved
    input_dir = Path(args.input_dir).expanduser().resolve()
    names = args.datasets or [p.name.removesuffix("_stats.npz") for p in sorted(input_dir.glob("*_stats.npz"))]
    return [(name, input_dir / f"{name}_stats.npz") for name in names]


def parse_layers(raw_layers: Sequence[int] | None, stats: Stats, all_layers: bool, layer_frame: pd.DataFrame) -> list[int]:
    if all_layers:
        return stats.layer_ids.tolist()
    if raw_layers:
        return [layer for layer in raw_layers if layer in stats.layer_ids]
    # Default report samples early/middle/late plus the most imbalanced layer.
    indexes = (0, len(stats.layer_ids) // 2, len(stats.layer_ids) - 1)
    selected = [int(stats.layer_ids[i]) for i in indexes]
    selected.append(int(layer_frame.loc[layer_frame.route_cv.idxmax(), "layer"]))
    return list(dict.fromkeys(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputs", nargs="+", help="Explicit NPZ paths. Dataset names are inferred from filenames.")
    parser.add_argument("--input-dir", default="/workspace/fd_afd_dev/offline_exp/runs/moe_stats_tp4_20260806")
    parser.add_argument("--datasets", nargs="+", help="Dataset basenames under --input-dir.")
    parser.add_argument("--output-dir", default="/workspace/fd_afd_dev/offline_exp/vis/output")
    parser.add_argument("--layers", nargs="+", type=int, help="Layer detail pages to render.")
    parser.add_argument("--all-layers", action="store_true", help="Render a detail page for every MoE layer.")
    parser.add_argument("--top-k", type=int, default=8, help="Model routing top-k, used to derive route share.")
    parser.add_argument("--closest-pairs", type=int, default=20, help="Pairs retained per layer in CSV.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip the combined multi-page PDF.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_style()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_specs = resolve_inputs(args)
    if not input_specs:
        raise SystemExit("No input NPZ files found")
    missing = [str(path) for _, path in input_specs if not path.is_file()]
    if missing:
        raise SystemExit("Missing input files:\n  " + "\n  ".join(missing))
    stats_list = [load_stats(path, name) for name, path in input_specs]
    layer_frames = [layer_metrics(stats) for stats in stats_list]
    expert_frames = [expert_metrics(stats, args.top_k) for stats in stats_list]
    pair_frames = [closest_pairs(stats, args.closest_pairs) for stats in stats_list]
    all_layers = pd.concat(layer_frames, ignore_index=True)
    all_experts = pd.concat(expert_frames, ignore_index=True)
    all_pairs = pd.concat(pair_frames, ignore_index=True)

    all_layers.to_csv(output_dir / "layer_summary.csv", index=False)
    all_experts.to_csv(output_dir / "expert_summary.csv", index=False)
    all_pairs.to_csv(output_dir / "closest_pairs.csv", index=False)
    candidates = (
        all_experts.sort_values(["dataset", "layer", "overlooked_score"], ascending=[True, True, False])
        .groupby(["dataset", "layer"], as_index=False)
        .head(16)
    )
    candidates.to_csv(output_dir / "overlooked_experts.csv", index=False)

    pdf_path = output_dir / "moe_stats_report.pdf"
    pdf = None if args.no_pdf else PdfPages(pdf_path)
    fig = global_overview(stats_list, layer_frames)
    fig.savefig(output_dir / "overview.png")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)
    fig = cross_dataset_view(stats_list, all_experts)
    fig.savefig(output_dir / "cross_dataset.png")
    if pdf:
        pdf.savefig(fig)
    plt.close(fig)

    rendered_pages = 2
    for stats, layers, expert_frame, pair_frame in zip(stats_list, layer_frames, expert_frames, pair_frames):
        dataset_dir = output_dir / stats.name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        fig = dataset_overview(stats, layers)
        fig.savefig(dataset_dir / "overview.png")
        if pdf:
            pdf.savefig(fig)
        plt.close(fig)
        rendered_pages += 1
        for layer_id in parse_layers(args.layers, stats, args.all_layers, layers):
            fig = layer_detail(stats, layer_id, expert_frame, pair_frame)
            fig.savefig(dataset_dir / f"layer_{layer_id:02d}.png")
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)
            rendered_pages += 1
    if pdf:
        info = pdf.infodict()
        info["Title"] = "All-expert MoE statistics report"
        info["Subject"] = "Expert distance, importance, routing load, and cross-dataset stability"
        pdf.close()

    print(f"Loaded {len(stats_list)} datasets: {', '.join(x.name for x in stats_list)}")
    print(f"Wrote {len(all_layers):,} layer rows and {len(all_experts):,} expert rows")
    print(f"Rendered {rendered_pages} report pages to {output_dir}")
    if pdf:
        print(f"PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
