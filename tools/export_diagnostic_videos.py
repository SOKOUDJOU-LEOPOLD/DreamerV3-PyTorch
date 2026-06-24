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

"""Offline diagnostic video export: load a checkpoint, sample real trajectories
from its associated (on-disk) replay buffer, and export Hook A (open-loop
world-model prediction) and Hook B (policy imagination rollout) videos.

Example:
    python tools/export_diagnostic_videos.py --env_name car_racing \
        --run_name baseline --checkpoint checkpoints_15000.ckpt --num_clips 2
"""

import os
import sys
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nnet.models.rl.dreamer.dreamerv3_diagnostics import export_open_loop_video, export_imagination_video


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/DreamerV3/dreamer_v3.py")
    parser.add_argument("--env_name", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--override_config", default="{}", help="JSON string, must match the run being loaded")
    parser.add_argument("--checkpoint", required=True, help="checkpoint filename inside the run's callback_path, e.g. checkpoints_15000.ckpt")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--num_clips", type=int, default=1)
    parser.add_argument("--context_frames_open_loop", type=int, default=5)
    parser.add_argument("--total_frames_open_loop", type=int, default=50)
    parser.add_argument("--context_frames_imagination", type=int, default=1)
    parser.add_argument("--img_steps_imagination", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Drive the existing config module exactly like main.py does -- this builds
    # the model, calls model.compile(), and model.set_replay_buffer(training_dataset)
    # (which must happen BEFORE model.load() below, since load() restores the
    # buffer's saved state_dict into the already-attached buffer object).
    os.environ["env_name"] = args.env_name
    os.environ["override_config"] = args.override_config
    if args.run_name:
        os.environ["run_name"] = args.run_name

    config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    model = config.model

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    checkpoint_path = os.path.join(config.callback_path, args.checkpoint)
    model.load(checkpoint_path, load_optimizer=False, verbose=True)
    model.eval()

    out_dir = args.out_dir or os.path.join("videos", "diagnostics", args.run_name or "default", args.env_name)
    tag = args.checkpoint.replace(".ckpt", "")

    # Sample one real batch from the restored replay buffer via its own collate_fn
    loader = torch.utils.data.DataLoader(
        dataset=model.replay_buffer,
        batch_size=args.num_clips,
        collate_fn=model.replay_buffer.collate_fn,
    )
    batch = next(iter(loader))["inputs"]
    batch = [t.to(device) for t in batch]

    for i in range(args.num_clips):
        p1 = export_open_loop_video(
            model, batch, tag="{}_clip{}".format(tag, i), out_dir=out_dir,
            context_frames=args.context_frames_open_loop, total_frames=args.total_frames_open_loop,
            fps=args.fps, batch_index=i
        )
        p2 = export_imagination_video(
            model, batch, tag="{}_clip{}".format(tag, i), out_dir=out_dir,
            context_frames=args.context_frames_imagination, img_steps=args.img_steps_imagination,
            fps=args.fps, batch_index=i
        )
        print("Wrote", p1)
        print("Wrote", p2)


if __name__ == "__main__":
    main()
