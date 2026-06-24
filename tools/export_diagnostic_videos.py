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

"""Offline diagnostic video export: load a checkpoint, roll out a fresh live
episode with its policy, and export Hook A (open-loop world-model prediction)
and Hook B (policy imagination rollout) videos.

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

from nnet.models.rl.dreamer.dreamerv3_diagnostics import export_open_loop_video, export_imagination_video, collect_live_episode


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
    parser.add_argument("--total_frames_open_loop", type=int, default=405)
    parser.add_argument("--context_frames_imagination", type=int, default=1)
    parser.add_argument("--img_steps_imagination", type=int, default=404)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--policy_mode", default=None, choices=[None, "sample", "mode"], help="defaults to the model's configured eval_policy_mode")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Drive the existing config module exactly like main.py does -- this builds
    # the model and calls model.compile(). model.set_replay_buffer(training_dataset)
    # still runs as part of importing the config, but we no longer rely on the
    # buffer for sampling -- diagnostic clips come from a fresh live rollout on
    # model.env_eval instead (collect_live_episode), which both removes the old
    # model.config.L=64 cap on clip length and sidesteps any replay buffer
    # save_state_dict corruption, since override_config below disables loading it.
    os.environ["env_name"] = args.env_name
    os.environ["override_config"] = args.override_config
    if args.run_name:
        os.environ["run_name"] = args.run_name

    config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    model = config.model

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = model.to(device)

    # This tool no longer samples from the replay buffer (see collect_live_episode
    # below), so skip restoring its state_dict -- avoids depending on the buffer's
    # on-disk save being intact, which it isn't for every checkpoint (see
    # DreamerV3ReplayBuffer.save()'s last_save_traj_index/traj_index race when a
    # step-triggered and epoch-triggered save land on the same step).
    model.config.load_replay_buffer_state_dict = False

    checkpoint_path = os.path.join(config.callback_path, args.checkpoint)
    model.load(checkpoint_path, load_optimizer=False, verbose=True)
    model.eval()

    # config.time_limit is an eval-time cap in raw env steps (e.g. car_racing's
    # default 1000 / action_repeat=4 = 250 agent steps), unrelated to the env
    # actually finishing (done=True). Raise it so a diagnostic clip isn't cut
    # short by this artifact -- real episode termination (done=True) still ends
    # collect_live_episode early regardless of this value.
    model.config.time_limit = max(model.config.time_limit, args.total_frames_open_loop * model.env_eval.action_repeat + model.env_eval.action_repeat)

    out_dir = args.out_dir or os.path.join("videos", "diagnostics", args.run_name or "default", args.env_name)
    tag = args.checkpoint.replace(".ckpt", "")

    for i in range(args.num_clips):
        # Fresh live episode per clip -- real frames/actions, length capped only
        # by how long the episode actually runs before done/time_limit.
        batch = collect_live_episode(model, num_frames=args.total_frames_open_loop, policy_mode=args.policy_mode)
        if batch[0].shape[1] < args.total_frames_open_loop:
            print("Clip {}: episode ended at {} frames (< requested {})".format(i, batch[0].shape[1], args.total_frames_open_loop))
        batch = [t.to(device) for t in batch]

        p1 = export_open_loop_video(
            model, batch, tag="{}_clip{}".format(tag, i), out_dir=out_dir,
            context_frames=args.context_frames_open_loop, total_frames=args.total_frames_open_loop,
            fps=args.fps, batch_index=0
        )
        p2 = export_imagination_video(
            model, batch, tag="{}_clip{}".format(tag, i), out_dir=out_dir,
            context_frames=args.context_frames_imagination, img_steps=args.img_steps_imagination,
            fps=args.fps, batch_index=0
        )
        print("Wrote", p1)
        print("Wrote", p2)


if __name__ == "__main__":
    main()
