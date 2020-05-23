import gym
import retro
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np

if _name_ == '__main__':
    env = retro.make("SpaceInvaders-Atari2600", state='Start')
    num_cpu = 7
    env = SubprocVecEnv([lambda: env for i in range(num_cpu)])

    model = A2C(CnnPolicy, env, learning_rate=0.0004, n_steps=25, verbose=1)
    model.learn(total_timesteps=250000)

    model.save("SpaceInvaders-Atari2600-new2")

    obs = env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
