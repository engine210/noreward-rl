import vizdoom as vzd
from collections import deque
import numpy as np
from PIL import Image
import moviepy.editor as mpy

class MyDoom():
    def __init__(self):
        self.obs_shape = (42, 42)
        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.maxFrames = True
        self.n = 4
        self.skip = 4
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
        self.ch_axis = -1
        self.scale = 1.0 / 255
        self.skip = 4
        self.images = []
        self.env = self.create_doom_game()

    def create_doom_game(self):
        game = vzd.DoomGame()
        game.set_doom_scenario_path("../wads/my_way_home_dense.wad")
        game.set_doom_map("map01")
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
        # game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.add_available_button(vzd.Button.TURN_LEFT)
        game.add_available_button(vzd.Button.TURN_RIGHT)
        game.add_available_button(vzd.Button.MOVE_FORWARD)
        game.set_episode_timeout(2100)
        #game.set_window_visible(True)
        game.set_window_visible(False)
        game.set_living_reward(0)
        game.set_mode(vzd.Mode.PLAYER)
        #game.set_console_enabled(True)
        game.init()
        return game

    def close(self):
        self.env.close()

    def step(self, action):
        reward = self.env.make_action(action)
        done = self.env.is_episode_finished()
        if done: return None, reward, done, None
        original_obs = self.env.get_state().screen_buffer
        obs = original_obs = np.moveaxis(original_obs, 0, -1)
        original_obs = np.array(Image.fromarray(original_obs).resize(self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)

        return self._observation(obs), reward, done, original_obs

    def skip_step(self, action):
        total_reward = 0
        for i in range(0, self.skip):
            obs, reward, done, original_obs = self.step(action)
            total_reward += reward
            if done:
                break
        self.images.append(original_obs)
        return obs, total_reward, done
    
    def _observation(self, obs):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def make_gif(self, fname, duration=5, true_image=True):
        images = self.images
        while images[-1] is None: images = images[:-1]
        def make_frame(t):
            try:
                x = images[int(len(images)/duration*t)]
            except:
                x = images[-1]

            if true_image:
                return x.astype(np.uint8)
            else:
                return ((x+1)/2*255).astype(np.uint8)

        clip = mpy.VideoClip(make_frame, duration=duration)
        clip.write_gif(fname, fps = len(images) / duration)

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        self.images = []
        self.env.new_episode()
        obs = self.env.get_state().screen_buffer
        obs = np.moveaxis(obs, 0, -1)
        obs = self._convert(obs)
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale

    def _convert(self, obs):
        self.obs_buffer.append(obs)
        if self.maxFrames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(Image.fromarray(intensity_frame).resize(
            self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame

    def _rgb2y(self, im):
        """Converts an RGB image to a Y image (as in YUV).

        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        # if len(im.shape) < 3:
        #     return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)