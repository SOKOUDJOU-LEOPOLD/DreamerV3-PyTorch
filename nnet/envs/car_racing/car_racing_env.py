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

# Other
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy") # headless pygame rendering, no GLX/X11 needed
import datetime
import numpy as np
import cv2

# PyTorch
import torch
import torchvision

# NeuralNets
from nnet.structs import AttrDict

# Gymnasium (gym==0.19.0's CarRacing needs pyglet/GLX which this server doesn't have;
# gymnasium's CarRacing renders through pygame and works fully headless)
import gymnasium.envs.box2d.car_racing as car_racing

class CarRacingEnv:

    """

    Version:
        gymnasium[box2d]==1.3.0

    Top-down procedurally generated race track, continuous control
    (steer, gas, brake). Episode terminates when the car leaves the
    playfield or finishes a lap.

    The policy's action space stays symmetric [-1, 1] for all 3 dims
    (matching this project's other continuous-action envs). Box2D's
    Car.gas() clips to [0, 1] internally but Car.brake() does not, so
    gas/brake are remapped from [-1, 1] to [0, 1] here before stepping
    the underlying env.

    """

    def __init__(self, img_size=(64, 64), action_repeat=4, history_frames=1, seed=None, episode_saving_path=None):

        # Env (instantiate directly, bypassing gym.make's registry wrappers, like AtariEnv does)
        self.env = car_racing.CarRacing(continuous=True, render_mode="rgb_array")

        # Params
        self.img_size = img_size
        self.action_repeat = action_repeat
        self.history_frames = history_frames

        # Action Space
        self.num_actions = 3
        self.action_size = (self.num_actions,)
        self.clip_low = -1
        self.clip_high = 1

        # Episode saving path
        self.episode_saving_path = episode_saving_path
        if self.episode_saving_path is not None:
            if not os.path.isdir(self.episode_saving_path):
                os.makedirs(self.episode_saving_path, exist_ok=True)

        # Seed
        self.seed(seed)

        # FPS
        self.fps = 50.0 / self.action_repeat

    def seed(self, seed):
        if seed:
            self.env.reset(seed=seed)

    def sample(self):

        action = torch.zeros(self.num_actions)
        action.uniform_(self.clip_low, self.clip_high)

        return action

    def preprocess(self, obs, reward, done):

        # Resize to img_size and to (C, H, W) uint8
        state = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        state = torch.tensor(state).permute(2, 0, 1) # uint8, mem efficient for buffer

        # Reward
        reward = torch.tensor(reward, dtype=torch.float32)

        # Done
        done = torch.tensor(done, dtype=torch.float32)

        # Is_last
        is_last = done

        return state, reward, done, is_last

    def reset(self):

        # Reset
        obs, _ = self.env.reset()
        state, _, _, _ = self.preprocess(obs, 0.0, False)

        # Episode video
        if self.episode_saving_path is not None:
            self.episode_video = []

        # Repeat history frames along channels
        if self.history_frames > 1:
            self.history = state.repeat(self.history_frames, 1, 1)
        else:
            self.history = state

        # Episode Score
        self.episode_score = 0.0

        # Reward
        reward = torch.tensor(0.0, dtype=torch.float32)

        # Done
        done = torch.tensor(False, dtype=torch.float32)

        # Is_last
        is_last = torch.tensor(False, dtype=torch.float32)

        # Is First
        is_first = torch.tensor(True, dtype=torch.float32)

        return AttrDict(state=self.history, reward=reward, done=done, is_first=is_first, is_last=is_last)

    def step(self, action):

        # Assert
        assert action.shape == self.action_size

        # Clip Actions to the policy's symmetric range
        action = action.clip(self.clip_low, self.clip_high)

        # Remap gas / brake from [-1, 1] to Box2D's expected [0, 1]
        env_action = action.clone()
        env_action[1] = (env_action[1] + 1.0) / 2.0
        env_action[2] = (env_action[2] + 1.0) / 2.0
        env_action = env_action.numpy().astype(np.float32)

        # Forward Env with action repeat
        reward = 0.0
        done = False
        for _ in range(self.action_repeat):

            # Env step
            obs, step_reward, step_terminated, step_truncated, _ = self.env.step(env_action)

            # Add to video
            if self.episode_saving_path is not None:
                self.episode_video.append(torch.tensor(obs))

            # Update reward / done
            reward += step_reward
            done = done or step_terminated
            if done:
                break

        # Preprocessing
        state, reward, done, is_last = self.preprocess(obs, reward, done)

        # Update Episode Score
        self.episode_score += reward.item()

        # Save Episode
        if done.item() and self.episode_saving_path is not None:

            # Stack video
            self.episode_video = torch.stack(self.episode_video, dim=0)

            # Datetime
            date_time_score = str(datetime.datetime.now()).replace(" ", "_") + "_" + str(self.episode_score)

            # Save Video
            torchvision.io.write_video(filename=os.path.join(self.episode_saving_path, "{}.mp4".format(date_time_score)), video_array=self.episode_video, fps=self.fps, video_codec="libx264")

        # Is First
        is_first = torch.tensor(False, dtype=torch.float32)

        # Concat history frames along channels
        if self.history_frames > 1:
            self.history = torch.cat([self.history[3:], state], dim=0)
        else:
            self.history = state

        return AttrDict(state=self.history, reward=reward, done=done, is_first=is_first, is_last=is_last)
