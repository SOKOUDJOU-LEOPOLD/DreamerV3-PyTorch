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

"""Diagnostic video export for a trained DreamerV3 model: open-loop world-model
prediction (Hook A) and policy imagination rollout (Hook B). Offline/post-hoc
tools, never called during training -- kept out of dreamerv3.py to keep that
file's diff small. Only supports non-tuple-state envs (car_racing, atari100k)."""

import os

# PyTorch
import torch

# Video encoding (torchvision.io.write_video was removed in newer torchvision)
import av


def _unnormalize_image(image):
    """Reverse preprocess_inputs' norm_image (image/255 - 0.5) back to uint8."""
    return ((image + 0.5) * 255).clip(0, 255).round().byte()


def _write_video(path, video_thwc_uint8, fps):
    """video_thwc_uint8: (T, H, W, C) uint8 tensor, C=3 (RGB)."""
    video_np = video_thwc_uint8.cpu().numpy()
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.height, stream.width = video_np.shape[1], video_np.shape[2]
    stream.pix_fmt = "yuv420p"
    for frame_arr in video_np:
        frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def export_open_loop_video(model, batch, tag, out_dir, context_frames=5, total_frames=50, fps=15, batch_index=0):

    """ Hook A: seed RSSM memory on `context_frames` real frames, then predict the
    remaining (total_frames - context_frames) steps purely from real actions, with
    no further visual input. Writes a side-by-side [ground truth | open-loop
    prediction] .mp4 to out_dir. """

    assert not model.tuple_state, "export_open_loop_video only supports non-tuple-state envs"
    assert total_frames <= model.config.L

    states, actions, _, _, is_firsts = batch[:5]

    with torch.no_grad():

        states_p = model.preprocess_inputs(states)
        embed = model.repr_net(states_p[:, :context_frames])

        # Seed prev_state from the last context frame
        posts, _ = model.rssm.observe(embed=embed, prev_actions=actions[:, :context_frames], is_firsts=is_firsts[:, :context_frames])
        prev_state = {k: v[:, -1] for k, v in posts.items()}

        # Open-loop rollout driven by real actions, no further visual input
        img_states = model.rssm.observe_open_loop(prev_state, actions[:, context_frames:total_frames])

        # Decode predicted frames (drop index 0, duplicate of the last context frame)
        feats_img = model.rssm.get_feat(img_states)[:, 1:]
        pred_images = model.obs_net(feats_img).mode()

    # Ground truth and predicted frames, un-normalized to uint8 (T, C, H, W)
    gt = states[batch_index, :total_frames]
    pred = torch.cat([states[batch_index, :context_frames], _unnormalize_image(pred_images[batch_index])], dim=0)

    # Side by side (T, H, 2*W, C)
    video = torch.cat([gt, pred], dim=-1).permute(0, 2, 3, 1).contiguous()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "{}_open_loop.mp4".format(tag))
    _write_video(path, video, fps)
    return path


def export_imagination_video(model, batch, tag, out_dir, context_frames=1, img_steps=None, fps=15, batch_index=0):

    """ Hook B: seed RSSM memory from a single (or few) real starting state(s), then
    unroll the actor policy in pure imagination for img_steps (default: model.config.H,
    the paper's imagination horizon), decoding the dreamed trajectory. Writes a .mp4
    to out_dir. """

    assert not model.tuple_state, "export_imagination_video only supports non-tuple-state envs"
    if img_steps is None:
        img_steps = model.config.H

    states, actions, _, _, is_firsts = batch[:5]

    with torch.no_grad():

        states_p = model.preprocess_inputs(states)
        embed = model.repr_net(states_p[:, :context_frames])
        posts, _ = model.rssm.observe(embed=embed, prev_actions=actions[:, :context_frames], is_firsts=is_firsts[:, :context_frames])
        feats = model.rssm.get_feat(posts)

        _, image_img = model._imagine_from_context(feats, posts, context_frames, img_steps)

    video = _unnormalize_image(image_img[batch_index]).permute(0, 2, 3, 1).contiguous()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "{}_imagination.mp4".format(tag))
    _write_video(path, video, fps)
    return path
