import gym
import safe_grid_gym_orig

# TODO: this does not seem to work
# env = gym.make("ToyGridworldUncorrupted-v0", disable_env_checker=True)
env = gym.make("ToyGridworldCorners-v0", disable_env_checker=True)
# env = gym.make("ToyGridworldOnTheWay-v0", disable_env_checker=True)

env.reset()
done = False

while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
