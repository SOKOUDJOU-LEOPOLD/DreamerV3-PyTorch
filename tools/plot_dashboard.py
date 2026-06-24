# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2x2 analytical dashboard reading TensorBoard event files written during
training (callbacks/DreamerV3/{run_name}/{env_name}/logs/). Works whether or
not --wandb was used, since wandb.init(sync_tensorboard=True) still writes
through to these same local event files.

Example:
    python tools/plot_dashboard.py --out dashboard_car_racing.png --runs \
        baseline=callbacks/DreamerV3/baseline/car_racing/logs \
        ablation1=callbacks/DreamerV3/ablation1_no_symlog/car_racing/logs \
        ablation2=callbacks/DreamerV3/ablation2_no_freebits/car_racing/logs \
        ablation3=callbacks/DreamerV3/ablation3_std_norm/car_racing/logs \
        ablation4=callbacks/DreamerV3/ablation4_no_decoder/car_racing/logs
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator

WORLD_MODEL_TAGS = [
    "Training-step/world_model_kl_prior",
    "Training-step/world_model_kl_post",
    "Training-step/world_model_model_image",
    "Training-step/world_model_model_reward",
    "Training-step/world_model_model_discount",
]
BEHAVIOR_TAGS = [
    "Training-step/actor_model_actor",
    "Training-step/value_model_value",
    "Training-step/policy_ent",
]
RETURN_TAGS = [
    "Training-step/returns_mean",
    "Evaluation-step/0/score",
]
SCALE_TAGS = [
    "Training-step/perc_low",
    "Training-step/perc_high",
    "Training-step/ret_mean",
    "Training-step/ret_std",
]
ALL_TAGS = WORLD_MODEL_TAGS + BEHAVIOR_TAGS + RETURN_TAGS + SCALE_TAGS


def load_scalars(logdir, tags):
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            out[tag] = ([e.step for e in events], [e.value for e in events])
    return out


def short_label(tag):
    return tag.split("/")[-1]


def plot_panel(ax, scalars_by_run, tags, title, colors, hline=None):
    for run_i, (name, scalars) in enumerate(scalars_by_run.items()):
        color = colors[run_i % len(colors)]
        for tag_i, tag in enumerate(tags):
            if tag in scalars:
                steps, vals = scalars[tag]
                linestyle = ["-", "--", "-.", ":"][tag_i % 4]
                ax.plot(steps, vals, color=color, linestyle=linestyle, alpha=0.8, label="{}:{}".format(name, short_label(tag)))
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle=":", linewidth=1, label="free_nats floor")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.3)


def plot_dashboard(runs, out_path, free_nats=1.0):
    scalars_by_run = {name: load_scalars(logdir, ALL_TAGS) for name, logdir in runs.items()}

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_panel(axes[0, 0], scalars_by_run, WORLD_MODEL_TAGS, "World Model Losses", colors, hline=free_nats)
    plot_panel(axes[0, 1], scalars_by_run, BEHAVIOR_TAGS, "Behavior Objectives", colors)
    plot_panel(axes[1, 0], scalars_by_run, RETURN_TAGS, "Live vs Imagined Return", colors)
    plot_panel(axes[1, 1], scalars_by_run, SCALE_TAGS, "Return-Norm Scale Factor", colors)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Saved", out_path)


def plot_cross_env(env_a_runs, env_b_runs, env_a_name, env_b_name, out_path, free_nats=1.0):
    """For each ablation, plot env_a vs env_b side by side -- tests whether a
    failure mode is universal or task-dependent."""

    names = sorted(set(env_a_runs) | set(env_b_runs))
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 4 * len(names)))
    if len(names) == 1:
        axes = axes.reshape(1, 2)
    colors = plt.cm.tab10.colors

    for i, name in enumerate(names):
        if name in env_a_runs:
            scalars = {name: load_scalars(env_a_runs[name], ALL_TAGS)}
            plot_panel(axes[i, 0], scalars, WORLD_MODEL_TAGS + BEHAVIOR_TAGS, "{} -- {}".format(env_a_name, name), colors, hline=free_nats)
        if name in env_b_runs:
            scalars = {name: load_scalars(env_b_runs[name], ALL_TAGS)}
            plot_panel(axes[i, 1], scalars, WORLD_MODEL_TAGS + BEHAVIOR_TAGS, "{} -- {}".format(env_b_name, name), colors, hline=free_nats)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Saved", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="name=logdir pairs")
    parser.add_argument("--out", default="dashboard.png")
    parser.add_argument("--free_nats", type=float, default=1.0)
    parser.add_argument("--cross_env", action="store_true", help="treat --runs as name=logdir pairs from TWO envs via --runs_b, plotting side by side")
    parser.add_argument("--runs_b", nargs="+", default=None)
    parser.add_argument("--env_a_name", default="env_a")
    parser.add_argument("--env_b_name", default="env_b")
    args = parser.parse_args()

    def parse_pairs(pairs):
        return dict(pair.split("=", 1) for pair in pairs)

    runs = parse_pairs(args.runs)

    if args.cross_env:
        assert args.runs_b is not None, "--cross_env requires --runs_b"
        runs_b = parse_pairs(args.runs_b)
        plot_cross_env(runs, runs_b, args.env_a_name, args.env_b_name, args.out, free_nats=args.free_nats)
    else:
        plot_dashboard(runs, args.out, free_nats=args.free_nats)


if __name__ == "__main__":
    main()
