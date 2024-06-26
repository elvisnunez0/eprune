from plotting.ckpt_metrics import get_accs_and_epochs, get_accs_during_training
from plotting.fisher import get_tr_fisher
from plotting.plot_style import (LINE_WIDTH, MPL_COLOR_CYCLE, PT_METRIC_COLOR,
                                 TR_FISHER_COLOR, plt)
from plotting.pt_metrics import get_pt_metrics
from plotting.utils import create_save_dir_from_path


def plot_training_accs(
    base_dir: str,
    fisher_path: str,
    save_path: str,
    title: str = None,
):
    tr_fisher = get_tr_fisher(fisher_path=fisher_path)[1:]
    train_accs = get_accs_during_training(base_dir=base_dir)[1:]

    plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    acc_color = MPL_COLOR_CYCLE[0]
    tr_fisher_color = MPL_COLOR_CYCLE[2]

    ax2.plot(tr_fisher, color=tr_fisher_color)

    ax1.plot(train_accs, color=acc_color, linewidth=LINE_WIDTH)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)", color=acc_color)

    ax2.set_ylabel("tr(F)", color=tr_fisher_color)

    ax1.grid(True, alpha=0.5)
    ax1.tick_params(axis="y", colors=acc_color)
    ax2.tick_params(axis="y", colors=tr_fisher_color)

    if title is not None:
        ax1.set_title(title)

    create_save_dir_from_path(save_path)
    plt.savefig(save_path)


def generic_acc_pt_metric_plot(
    ckpts_dir: str,
    baseline_dir: str,
    pt_metrics_dir: str,
    save_path: str,
    metric_name: str = "fisher",
    title: str = None,
):
    ckpt_nums, best_accs, best_baseline_acc = get_accs_and_epochs(
        ckpts_dir=ckpts_dir,
        baseline_dir=baseline_dir,
    )

    if metric_name == "fisher":
        pt_metrics = get_tr_fisher(fisher_path=pt_metrics_dir)[1:]
        pt_metrics_color = TR_FISHER_COLOR
        ylabel = "tr(F)"
    else:
        pt_metrics = get_pt_metrics(pt_metrics_dir=pt_metrics_dir)
        pt_metrics_color = PT_METRIC_COLOR
        ylabel = r"$KL_{Uniform}$"

    plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    acc_color = MPL_COLOR_CYCLE[0]
    baseline_acc_color = "black"

    ax2.plot(pt_metrics, color=pt_metrics_color)

    ax1.plot(ckpt_nums, best_accs, "o-", color=acc_color, linewidth=LINE_WIDTH)
    ax1.plot(
        [best_baseline_acc] * len(pt_metrics),
        color=baseline_acc_color,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Baseline",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)", color=acc_color)

    ax2.set_ylabel(ylabel, color=pt_metrics_color)

    ax1.legend(loc="best")

    ax1.grid(True, alpha=0.5)
    ax1.tick_params(axis="y", colors=acc_color)
    ax2.tick_params(axis="y", colors=pt_metrics_color)

    if title is not None:
        ax1.set_title(title)

    create_save_dir_from_path(save_path)

    plt.tight_layout()
    plt.savefig(save_path)


def generic_acc_fisher_plot(
    ckpts_dir: str,
    baseline_dir: str,
    fisher_path: str,
    save_path: str,
    title: str = None,
):
    ckpt_nums, best_accs, best_baseline_acc = get_accs_and_epochs(
        ckpts_dir=ckpts_dir,
        baseline_dir=baseline_dir,
    )

    tr_fisher = get_tr_fisher(fisher_path=fisher_path)[1:]

    plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    acc_color = MPL_COLOR_CYCLE[0]
    baseline_acc_color = "black"

    ax2.plot(tr_fisher, color=TR_FISHER_COLOR)

    ax1.plot(ckpt_nums, best_accs, "o-", color=acc_color, linewidth=LINE_WIDTH)
    ax1.plot(
        [best_baseline_acc] * len(tr_fisher),
        color=baseline_acc_color,
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Baseline",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)", color=acc_color)

    ax2.set_ylabel("tr(F)", color=TR_FISHER_COLOR)

    ax1.legend(loc="best")

    ax1.grid(True, alpha=0.5)
    ax1.tick_params(axis="y", colors=acc_color)
    ax2.tick_params(axis="y", colors=TR_FISHER_COLOR)

    if title is not None:
        ax1.set_title(title)

    create_save_dir_from_path(save_path)

    plt.tight_layout()
    plt.savefig(save_path)


def generic_entropy_fisher_plot(
    entropy_path: str,
    fisher_path: str,
    save_path: str,
    title: str = None,
):
    tr_fisher = get_tr_fisher(fisher_path=fisher_path)[1:]
    entropies = get_pt_metrics(pt_metrics_dir=entropy_path)[1:]

    plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    acc_color = MPL_COLOR_CYCLE[0]
    baseline_acc_color = "black"

    ax2.plot(tr_fisher, color=TR_FISHER_COLOR)

    ax1.plot(entropies, color=acc_color, linewidth=LINE_WIDTH)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Entropy H(Y|X)", color=acc_color)

    ax2.set_ylabel("tr(F)", color=TR_FISHER_COLOR)

    ax1.legend(loc="best")

    ax1.grid(True, alpha=0.5)
    ax1.tick_params(axis="y", colors=acc_color)
    ax2.tick_params(axis="y", colors=TR_FISHER_COLOR)

    if title is not None:
        ax1.set_title(title)

    create_save_dir_from_path(save_path)
    plt.savefig(save_path)
