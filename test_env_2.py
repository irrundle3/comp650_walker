import gymnasium as gym

from stable_baselines3 import PPO
import math
import pygame
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class WalkerProto(gym.Env):
    MOVE_ANGLE_NOISE = 0.0
    MOVE_DIST_NOISE = 0.0
    MOVE_DIST_MEAN = 1.0
    MAX_DIST_BIAS = 0.0
    MAX_ANGLE_BIAS = 0.0
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array"
        ],
        "render_fps": 10
    }
    
    def __init__(self, render_mode = None):
        self.grid_size = 64
        self.window_size = 512
        self.max_start_distance = 5
        self.num_steps = 0
        self._agent_location = None
        self._target_location = None
        self.clock = None
        self.window = None
        self.num_steps = 0
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.grid_size, shape=(2,), dtype=float),
                "target": spaces.Box(0, self.grid_size, shape=(2,), dtype=float),
            }
        )
        
        self.action_space = spaces.Discrete(5)
        self.action_distance_biases = None
        self.action_angle_biases = None
        self.render_mode = render_mode
        self.reward = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reward = 0
        self.num_steps = 0
        self._agent_location = self.np_random.random(size=2) * self.grid_size
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.clip(self._target_location + (self.np_random.random(size=2) - 0.5) * self.max_start_distance / 2, 0, self.grid_size)
        self.action_distance_biases = (self.np_random.random(size=4) - 0.5) * 2 * WalkerProto.MAX_DIST_BIAS
        self.action_angle_biases = (self.np_random.random(size=4) - 0.5) * 2 * WalkerProto.MAX_ANGLE_BIAS
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        direction = self._generate_action_delta(action)
        self._agent_location = np.clip(
        	self._agent_location + direction, 0, self.grid_size
    	)
        self.num_steps += 1
        new_distance = self._get_info()["distance"]
   	 
    	# # Scaled distance reward
    	# reward = -new_distance / self.grid_size
        reward = 0

    	# Check if the goal is reached
        goal_reached = new_distance < 1  # Assuming a threshold of 0.05 units to consider the goal reached
        if goal_reached:
            reward += 1000  # Large positive reward for reaching the goal
        
    	# Small penalty for each step to encourage efficiency
        reward -= 0.1
        terminated = goal_reached
        
        if self.num_steps > self.grid_size * 2:
            terminated = True
        
        self.reward += reward
        observation = self._get_obs()
        info = self._get_info()
	
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
            
    def render(self):
        if self.render_mode in ("rgb_array", "human"):
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.grid_size # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self._target_location * pix_square_size,
            pix_square_size * 1,
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._agent_location * pix_square_size,
            pix_square_size * 1,
        )

        # Finally, add some gridlines
        for x in range(0, self.grid_size + 1, 10):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
            
        font = pygame.font.Font(pygame.font.get_default_font(), 24)
        text = font.render(str(self.reward), True, (255, 0, 0))
        rect = text.get_rect()
        rect.center = (self.window_size // 2, self.window_size - 30)
        canvas.blit(text, rect)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
    def _generate_action_delta(self, action):
        if action == 4:
            return np.array([0.0,0.0])
        angle = math.pi / 2 * action # Angle in radians before noise
        angle = np.random.normal(loc = angle, scale = WalkerProto.MOVE_ANGLE_NOISE) + self.action_angle_biases[action]
        dist = np.random.normal(loc = WalkerProto.MOVE_DIST_MEAN, scale = WalkerProto.MOVE_DIST_NOISE) + self.action_distance_biases[action]
        return np.array([math.cos(angle) * dist, math.sin(angle) * dist])
        
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=2
            )
        }

train = False
name = "walker_2d_4M"
env = WalkerProto(render_mode=("rgb_array" if train else "human")) 

if train:
    print("training...")
    model = PPO("MultiInputPolicy", env, ent_coef=0.01, verbose=1)
    # model = PPO.load(name, env=env)
    model.learn(total_timesteps=2_000_000)
    for i in range(10):
        env.max_start_distance += 2
        model.learn(total_timesteps=200_000)
    model.save(name)
else:
    env.max_start_distance = 25

print("Done training.")

model = PPO.load(name, env=env)

# env = WalkerProto(render_mode="human") 
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
