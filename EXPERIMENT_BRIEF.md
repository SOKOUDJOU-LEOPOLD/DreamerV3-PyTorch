# DreamerV3 Diagnostic & Ablation Experiments — Full Context Brief

## Who you are talking to / why this document exists

I (the user) am studying the DreamerV3 paper (Hafner et al., "Mastering Diverse Domains through World Models," arXiv 2301.04104) by building diagnostic tooling and ablation experiments against `DreamerV3-PyTorch`, a from-scratch PyTorch reimplementation of the paper (forked at `github.com/SOKOUDJOU-LEOPOLD/DreamerV3-PyTorch`, originally by Maxime Burchi). The goal is a presentation: show, empirically, what each of the paper's claimed "robustness techniques" actually does by breaking each one on purpose and visualizing the failure mode, on two environments. This document is a full handoff brief for a fresh agent with no memory of the conversation that produced it — read it top to bottom before doing anything.

## The two target environments

- **`car_racing`** — a new environment I had a previous Claude session integrate into this project (it did not exist before). Top-down procedurally generated race track, continuous control (steer/gas/brake), pixel observations. Wraps `gymnasium`'s `CarRacing` (not the old `gym==0.19.0` version bundled with the project, which needs a real X server/GLU library; `gymnasium`'s renders headlessly via `pygame`). Code lives at `nnet/envs/car_racing/car_racing_env.py`.
- **`atari100k-breakout`** — already existed in the project (the standard Atari 100k sample-efficiency benchmark), but required fixing: the pinned `atari-py==0.2.6` dependency is abandoned and doesn't build on modern Python/setuptools. It was replaced with a compatibility shim (`atari_py` module backed by `ale-py`, the actively maintained ALE bindings) — **this shim is NOT tracked in git**, it lives only inside the Python environment's `site-packages`, so it must be manually recreated on any new machine (full file content is in the migration section below).

## The original ask (paraphrased from my own spec)

I wrote a "Master Specification Sheet" asking for:
1. Train DreamerV3 at the paper's official "Large" (~77M parameter) model size, no downscaling, imagination horizon locked to 15, 4-8 parallel env workers.
2. Two diagnostic video hooks: **Hook A** (open-loop world-model prediction — seed on real frames, predict forward using only real recorded actions, no further visual input, compare against ground truth) and **Hook B** (policy imagination rollout — start from one real state, let the actor dream forward for 15 steps purely in imagination).
3. A baseline run with every robustness technique intact, with Hook A/B videos captured at early/mid/late training checkpoints to show evolution.
4. A 4-way ablation matrix, each ablation removing exactly one robustness technique, early-stopped (30-90 min) the moment a visible failure signature appears:
   - **Ablation 1**: no symlog + twohot discrete regression → plain MSE on raw targets (expect: loss spikes/instability on large rewards)
   - **Ablation 2**: no free-bits / KL floor → symmetric KL weighting (expect: posterior collapse or runaway KL)
   - **Ablation 3**: no percentile return normalization → std-based normalization instead (expect: noisier advantage estimates, degraded policy)
   - **Ablation 4**: no reconstruction/decoder loss (expect: degraded/uninformative learned representations)
5. A 2x2 analytical dashboard (world model losses, behavior objectives, live-vs-imagined return, return-normalization scale factor).

## Decisions made along the way (don't re-litigate these)

- **Both environments get the full ablation matrix**: 2 baselines + 4 ablations × 2 envs = **10 total training runs**.
- **`atari100k` stays at `num_envs=1`, NOT vectorized** even though the original spec asked for 4-8 parallel workers generally — that benchmark's entire point is comparability to the literature's "100k interactions" sample budget; vectorizing it would invalidate that comparison. Only `car_racing` is vectorized (`num_envs=4`, already the case).
- **Single config-driven git branch** (`diagnostics-and-ablations`, branched off `master`) — no per-ablation branches. Every ablation is just a different `override_config` JSON value on the same code.
- **Budget is tight** (~1-2 days, presentation soon) — "baseline" means a concrete, bounded step count, not literal convergence.

## What has actually been built (all done, committed, and pushed to GitHub)

Repo: `github.com/SOKOUDJOU-LEOPOLD/DreamerV3-PyTorch`. Branch: `diagnostics-and-ablations` (10 commits, branched off `master` which has the CarRacing integration). Both branches are pushed — `git clone` + `git checkout diagnostics-and-ablations` gets everything.

1. **Bugfix** (`nnet/models/rl/dreamer/dreamerv3.py`): `override_config={"model_size":"L"}` used to be a silent no-op — the network-dimension cascade ran *before* the override loop, so overriding `model_size` only relabeled it without changing actual capacity. Fixed by moving the cascade to run after the override loop. Verified: car_racing at default "S" = 19.1M params, with `"model_size":"L"` override = 81.1M params (matches the paper's ~77M Large config).
2. **Hook A** (`nnet/modules/dreamer/v3/rssm.py`): new method `RSSM.observe_open_loop(prev_state, actions)` — loops the existing `forward_img` primitive over a *given* action sequence instead of policy-sampled actions (the existing `imagine()` only supports the latter). Verified numerically identical to a manual `forward_img` loop.
3. **Hook B** (`nnet/models/rl/dreamer/dreamerv3.py`): the existing `log_figure()` method already did 90% of this (seed context → imagine via policy → decode), just as a static TensorBoard image grid. Extracted its imagination logic into a reusable helper `_imagine_from_context(...)`; `log_figure` now calls it (verified bit-identical output, zero behavior change to existing training-time visualization).
4. **Diagnostics module** (`nnet/models/rl/dreamer/dreamerv3_diagnostics.py`, new file): `export_open_loop_video` (Hook A — writes a side-by-side [ground truth | prediction] `.mp4`) and `export_imagination_video` (Hook B — writes the actor's pure dreamed rollout as `.mp4`). Uses `PyAV` directly for encoding, **not** `torchvision.io.write_video`, which was removed in the installed torchvision (0.27.1) — note this also means the project's *existing* real-episode video saving in `atari_env.py`/`car_racing_env.py` is currently broken on this torchvision version, a separate pre-existing issue, out of scope.
5. **CLI script** (`tools/export_diagnostic_videos.py`, new file): loads a checkpoint (mirroring `main.py`'s construction order — `model.set_replay_buffer()` must happen before `model.load()`, since `load()` restores the buffer's saved state into the already-attached buffer object), samples a real batch via the replay buffer's own `collate_fn`, runs both hooks. Verified end-to-end against a real checkpoint with a populated replay buffer.
6. **Ablation 1 code** (`nnet/modules/dreamer/v3/reward_network.py`, `value_network.py`): new constructor flag `use_symlog_twohot=True` (default = current/paper behavior). When `False`: 1-unit output head instead of 255-bin categorical, returns `MSEDist` (already exists, already used for pixel reconstruction) on the raw untransformed target instead of `SymLogDiscreteDist`. Wired via new config key `ablation_no_symlog_twohot=False`.
7. **Ablation 3 code** (`nnet/models/rl/dreamer/dreamerv3.py`, `update_perc`/`get_perc`): new config key `return_norm_method="percentile"` (default, unchanged) vs `"std"` — new `ret_mean`/`ret_std` buffers, EMA-tracked, computed explicitly in fp32 to avoid a precision artifact being mistaken for the ablation's real effect. Verified both branches against manual reference calculations; default path is bit-identical to pre-change behavior.
8. **Ablations 2 and 4 needed zero new code** — `free_nats` and `loss_decoder_scale` were already plain, ungated config values.
9. **Dashboard** (`tools/plot_dashboard.py`, new file): reads TensorBoard event files directly (works with or without `--wandb`, since `wandb.init(sync_tensorboard=True)` still writes the same local event files) using the confirmed exact tag convention (`Training-step/world_model_kl_prior`, `Training-step/returns_mean`, `Evaluation-step/0/score`, etc.). Four panels matching the original spec exactly, plus a bonus `--cross_env` mode comparing the same ablation across both environments side by side. Verified against synthetic-but-real-tag-format event files.

**Verification status**: every individual piece above is unit-tested. Additionally, all 10 final run configs (baseline + 4 ablations × 2 envs, all at real "L" 77M-parameter size) were validated by directly running the full world-model/actor/value loss computation (forward **and** backward pass) — all produce finite losses with no exceptions. **What has never been tested: actually running a real multi-step training loop end-to-end on a GPU.** That's the entire reason for the server migration below.

## Why we're migrating servers

The original machine (`csce-chzhang.engr.tamu.edu`) can enumerate its 4 H100 GPUs fine (`torch.cuda.is_available()` → True, correct device count/names) but **cannot create an actual CUDA context** — every attempt to move a tensor to `cuda:0` fails with `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable` (`cudaErrorDevicesUnavailable`). This was extensively diagnosed and the following were ruled out: stuck/orphaned processes (nvidia-smi shows 0% util, 0MiB used, no processes on any GPU), pending GPU resets/ECC errors, MPS daemon conflicts, `systemd-logind` session/linger issues (enabled linger, no change), and — critically — packaging (tested both the pip-installed venv's PyTorch AND a completely fresh Miniconda-installed PyTorch in an isolated environment; both hit the identical error). A server maintainer suggested conda specifically; that was tested and disproven. The likely remaining cause is a cgroup/eBPF-level device restriction or similar that requires root to diagnose, which we don't have. Conclusion: move to a different machine rather than keep debugging blind.

## The new machine: TAMU HPRC "FASTER" cluster

Logged in as `lsokoudj@faster2`. This is a real SLURM-managed HPC cluster (confirmed via the sudo-denial message referencing `hprc.tamu.edu`), fundamentally different from the old single shared box:
- `(base)` conda environment is active by default — HPRC's standard convention.
- No sudo, as expected/intended on HPC (don't try to work around this — it's by design, and is **not** the same kind of "weird restriction" as the old server's CUDA issue).
- `nvidia-smi` fails with "couldn't communicate with the NVIDIA driver" on the login node — this is very likely because **login nodes have no GPU hardware at all** on HPC clusters; GPUs only exist on compute nodes, accessed by submitting a job through SLURM (`sbatch`/`salloc`/`srun`), not by running things directly in the login shell.
- **Not yet confirmed**: whether SLURM is present and what the GPU partition is named. The immediate next diagnostic step (not yet done as of this brief) is:
  ```bash
  hostname
  which sbatch squeue sinfo module
  module avail 2>&1 | grep -iE "cuda|gpu"
  sinfo -o "%P %G %N"
  ```
  This tells you the GPU partition name and whether a `module load cuda` (or similar) step is needed before PyTorch/CUDA will work. Once known, GPU work happens via an SBATCH script or an interactive `salloc` allocation — **not** directly on the login node.

## Migration procedure (do this once SLURM/GPU access is confirmed)

**Phase 1 — Get the code:**
```bash
git clone https://github.com/SOKOUDJOU-LEOPOLD/DreamerV3-PyTorch.git
cd DreamerV3-PyTorch
git checkout diagnostics-and-ablations
```

**Phase 2 — Python environment** (conda is already the convention here, unlike the old server):
```bash
conda create -y -n dreamer python=3.12
conda activate dreamer
pip install -r requirements.txt   # skip torchtext if it errors -- unused in this codebase, deprecated upstream, not needed
```
Verify CUDA works on an actual GPU allocation (not the login node) before going further — this is the whole point of the move:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.zeros(1).cuda())"
```

**Phase 3 — Recreate the Atari shim (NOT in git, manual every time):**
```bash
pip install ale-py opencv-python-headless

# gym 0.19.0 has a setup.py bug (invalid version specifier) that breaks on modern setuptools -- patch it
curl -fsSL -o /tmp/gym.tar.gz "$(curl -s https://pypi.org/pypi/gym/0.19.0/json | python3 -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('.tar.gz')][0])")"
tar xzf /tmp/gym.tar.gz -C /tmp
sed -i 's/"opencv-python>=3\."/"opencv-python>=3.0"/' /tmp/gym-0.19.0/setup.py
pip install /tmp/gym-0.19.0
```
Then create `<site-packages>/atari_py/__init__.py` (find the path via `python3 -c "import site; print(site.getsitepackages()[0])"`) with this exact content:
```python
import ale_py
import ale_py.roms as _roms


def get_game_path(game):
    return str(_roms.get_rom_path(game))


def _to_str_key(key):
    return key.decode("utf-8") if isinstance(key, bytes) else key


class ALEInterface:

    def __init__(self):
        self._ale = ale_py.ALEInterface()

    def setFloat(self, key, value):
        self._ale.setFloat(_to_str_key(key), value)

    def setInt(self, key, value):
        self._ale.setInt(_to_str_key(key), value)

    def setBool(self, key, value):
        self._ale.setBool(_to_str_key(key), value)

    def setString(self, key, value):
        self._ale.setString(_to_str_key(key), value)

    def loadROM(self, path):
        self._ale.loadROM(path)

    def getAvailableModes(self):
        return list(self._ale.getAvailableModes())

    def setMode(self, mode):
        self._ale.setMode(mode)

    def getAvailableDifficulties(self):
        return list(self._ale.getAvailableDifficulties())

    def setDifficulty(self, difficulty):
        self._ale.setDifficulty(difficulty)

    def getLegalActionSet(self):
        return [a.value for a in self._ale.getLegalActionSet()]

    def getMinimalActionSet(self):
        return [a.value for a in self._ale.getMinimalActionSet()]

    def getScreenDims(self):
        height, width = self._ale.getScreenDims()
        return (width, height)

    def getRAMSize(self):
        return self._ale.getRAMSize()

    def getRAM(self, ram):
        self._ale.getRAM(ram)

    def act(self, action):
        return self._ale.act(action)

    def game_over(self):
        return self._ale.game_over()

    def lives(self):
        return self._ale.lives()

    def getScreenRGB2(self, *args):
        return self._ale.getScreenRGB(*args)

    def getScreenGrayscale(self, *args):
        return self._ale.getScreenGrayscale(*args)

    def reset_game(self):
        self._ale.reset_game()

    def cloneState(self):
        return self._ale.cloneState()

    def cloneSystemState(self):
        return self._ale.cloneSystemState()

    def restoreState(self, state):
        self._ale.restoreState(state)

    def restoreSystemState(self, state):
        self._ale.restoreSystemState(state)

    def encodeState(self, state):
        return state.serialize()

    def decodeState(self, data):
        return ale_py.ALEState(data)

    def deleteState(self, state):
        pass
```
Verify: `env_name=atari100k-breakout python3 main.py -c configs/DreamerV3/dreamer_v3.py -m pass --cpu`

**Phase 4 — CarRacing**: already covered by `requirements.txt` (`gymnasium[box2d]` pulls a prebuilt `box2d` wheel, no `swig` build needed). Verify: `env_name=car_racing python3 main.py -c configs/DreamerV3/dreamer_v3.py -m pass --cpu`

**Phase 5 — Confirm the model_size bugfix is active:**
```python
import nnet
m = nnet.models.rl.dreamer.DreamerV3(env_name='car_racing', override_config={'model_size':'L'})
print(sum(p.numel() for p in m.parameters()))  # expect ~81M
```

## The actual experiment run plan (once migration is verified)

10 runs total. Each uses `env_name`, `run_name`, and `override_config` env vars feeding into `configs/DreamerV3/dreamer_v3.py`, which already supports all of this with zero further code changes:

| # | env_name | run_name | override_config |
|---|---|---|---|
| 1 | car_racing | baseline | `{"model_size":"L","epochs":3}` |
| 2 | atari100k-breakout | baseline | `{"model_size":"L","epochs":3}` |
| 3 | car_racing | ablation1_no_symlog | `{"model_size":"L","ablation_no_symlog_twohot":true,"epochs":10}` |
| 4 | car_racing | ablation2_no_freebits | `{"model_size":"L","free_nats":0.0,"epochs":10}` |
| 5 | car_racing | ablation3_std_norm | `{"model_size":"L","return_norm_method":"std","epochs":10}` |
| 6 | car_racing | ablation4_no_decoder | `{"model_size":"L","loss_decoder_scale":0.0,"epochs":10}` |
| 7-10 | atari100k-breakout | (same 4 ablation run_names) | (same 4 override_configs) |

Recommended flags on every run: `--saving_period_step 500 --log_figure_period_step 500` (frequent checkpoints/TensorBoard images for monitoring and for early/mid/late video export later). Baselines have a tight `epochs` cap so they stop on their own; ablations have a generous safety-net cap (10 epochs) but the actual intent is to **watch** (TensorBoard, or the existing `log_figure` images) and stop manually (Ctrl+C, or cancel the SLURM job) the moment a clear failure signature appears — that's the whole point of the ablation, not running it to completion.

On this cluster, each of these 10 runs should become a SLURM job (`sbatch` script or interactive `salloc`) requesting one GPU, rather than a backgrounded shell process — adapt the env-var-prefixed command above into whatever SBATCH script format this cluster expects once the partition name is known.

**After each run** (or while ablations are paused for inspection), use `tools/export_diagnostic_videos.py` against specific checkpoints to get the Hook A/B videos, and `tools/plot_dashboard.py` against the run's TensorBoard logdir (`callbacks/DreamerV3/{run_name}/{env_name}/logs/`) for the 2x2 dashboard.

## Final deliverables for the presentation

- Hook A and Hook B videos at early/mid/late checkpoints for both baselines (showing the model's predictive/imaginative capability improving over training).
- Hook A and Hook B videos for each ablation at the point its failure became visible.
- **Per-step loss curves** for every run (world model losses — `kl_prior`, `kl_post`, `model_image`, `model_reward`, `model_discount` — plus actor/value losses and policy entropy). Logged automatically by the existing training loop to TensorBoard (`Training-step/...` tags); `tools/plot_dashboard.py`'s panels [0,0] and [0,1] plot these directly, overlaying baseline against each ablation.
- **Learning curves** for every run: live environment return (from periodic evaluation episodes) overlaid against the world model's imagined return estimate, over training steps — `tools/plot_dashboard.py`'s panel [1,0]. This is also where you'd see whether an ablation degrades actual task performance, not just internal losses.
- Per-environment 2x2 dashboards (baseline + all 4 ablations overlaid) combining all of the above plus the return-normalization scale factor panel.
- A cross-environment comparison (`--cross_env` dashboard mode) showing whether each ablation's failure is universal or task-dependent — directly testing the paper's own claim that "techniques are critical on a subset of tasks but may not affect performance on other tasks."
