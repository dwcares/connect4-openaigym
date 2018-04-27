import gym
import connect4game
env = gym.make('Connect4Game-v0')
print(env.action_space)
print(env.observation_space)

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action