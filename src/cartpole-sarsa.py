import gymnasium as gym
import numpy as np
import math
from random import random

# A state s (a.k.a. observation) has cart position, cart velocity, pole angle,
# and pole angular velocity.
# States are converted from continuous ranges into discrete numbers.  E.g. the
# cart position has a range of [-2.4, 2.4], which gets converted into a discrete
# number in the range [0, STATE_BUCKETS).
STATE_BUCKETS = 20
CART_POS_MIN = -2.4
CART_POS_MAX = 2.4

CART_VEL_MIN = -4
CART_VEL_MAX = 4

POLE_ANGLE_MIN = -.2095
POLE_ANGLE_MAX = .2095

POLE_VEL_MIN = -3
POLE_VEL_MAX = 3

# How many episodes to train on.  With more episodes, the model is more likely
# to converge on a solution, but it takes longer.
NUM_EPISODES = 30000

# The learning rate, usually denoted α.  Reward predictions are multipled by
# this number (lower learns slower).
LEARN_RATE = .1

# Using decaying ε-greedy for exploration.  The probability of choosing a random
# action decays from 1 to 0 over NUM_EPISODES * E_DECAY.
E_DECAY = .75

# Usually denoted γ, this determines how much to favor future rewards over
# immediate ones.  A value of 0 makes the agent only consider immediate rewards
# (myopic).
DISCOUNT = .99

# When the model achieves this average reward, training is done.
MEAN_REWARD_EPISODES = 100
MEAN_REWARD_TARGET = 495

# Number of episodes to run to show off the model after training is done.
TEST_EPISODES = 10

"""
Train, filling out the quality table.
"""
def train(env, q):
  episode = 0
  episode_rewards = []
  mean_reward = 0.0

  while episode < NUM_EPISODES and mean_reward < MEAN_REWARD_TARGET:
    episode += 1
    episode_over = False
    total_reward = 0

    # State (s0), made discrete, and action (a0).
    s0, _ = env.reset()
    s0 = make_discrete_state(s0)
    a0 = get_action(s0, episode, q)

    while not episode_over:
      # s1, a1: new state and next action after applying action a0.
      # r: +1 reward for each step the pole stays upright.
      # terminated: True if the pole falls too far or the cart goes off screen.
      # truncated: True if the time limit is reached (500 steps, i.e. mastery).
      s1, r, terminated, truncated, _ = env.step(a0)
      s1 = make_discrete_state(s1)
      a1 = get_action(s1, episode, q)

      # Update q in the direction of the error.  The expected/predicted reward
      # for transitioning from s0 to s1 is the predicted reward for the current
      # state/action less the future reward prediction.  The actual reward is r.
      #
      # Expected: q(s,a) - q(s',a')
      # Actual: r
      # Error: r - [q(s,a) - q(s',a')] = r + q(s',a') - q(s,a)
      #
      # The predicted future reward q(s',a') is discounted so that the immediate
      # reward matters more than future rewards, and the value function q(s,a)
      # (predicted reward) is updated in the direction of the error using
      # LEARN_RATE.
      q[s0][a0] += LEARN_RATE * (r + DISCOUNT * q[s1][a1] - q[s0][a0])

      total_reward += r
      episode_over = terminated or truncated

      s0 = s1
      a0 = a1

    episode_rewards.append(total_reward)
    mean_reward = np.mean(episode_rewards[-MEAN_REWARD_EPISODES:])

    if episode % 1000 == 0:
      print(f"Episode {episode} finished. {MEAN_REWARD_EPISODES}-episode mean reward: {mean_reward}")

"""
Test the quality table (full greedy action choice).
"""
def test(env, q):
  episode = 0

  while episode < TEST_EPISODES:
    episode += 1

    s, _ = env.reset()
    s = make_discrete_state(s)

    episode_over = False
    total_reward = 0

    while not episode_over:
      a = np.argmax(q[s])
      s, r, terminated, truncated, _ = env.step(a)
      s = make_discrete_state(s)

      total_reward += r
      episode_over = terminated or truncated

    print(f"Episode {episode} finished with total reward: {total_reward}.")

"""
Clamp val to range [val_min, val_max].
"""
def clamp(val, val_min, val_max):
  return max(val_min, min(val, val_max))

"""
Convert a continuous value val in the range [val_min, val_max] to a discrete
number in the range [0, size).  ("Digitize" in a linear space.)
"""
def make_discrete(val, val_min, val_max, size):
  # Move val to [0, val_max - val_min].
  translated = val - val_min
  scaled = translated * size / (val_max - val_min)
  discrete = clamp(math.floor(scaled), 0, size - 1)

  return discrete

"""
Convert a state with continuous numbers to an state with discrete
numbers, i.e. make_discrete on each observed value.
"""
def make_discrete_state(state):
  return (
    make_discrete(state[0], CART_POS_MIN, CART_POS_MAX, STATE_BUCKETS),
    make_discrete(state[1], CART_VEL_MIN, CART_VEL_MAX, STATE_BUCKETS),
    make_discrete(state[2], POLE_ANGLE_MIN, POLE_ANGLE_MAX, STATE_BUCKETS),
    make_discrete(state[3], POLE_VEL_MIN, POLE_VEL_MAX, STATE_BUCKETS)
  )

"""
Pick an action using decaying ε-greedy.  The decay is linear, decaying from
1 (100% random) to 0 (always use the quality table) over
NUM_EPISODES * E_DECAY episodes.
"""
def get_action(state, episode, q):
  prob = episode * -1 / (NUM_EPISODES * E_DECAY) + 1

  if random() < prob or np.sum(q[state]) == 0:
    action = env.action_space.sample()
  else:
    action = np.argmax(q[state])

  return action

if __name__ == '__main__':
  env = gym.make("CartPole-v1")

  # The "quality table", q.  A state, made discrete, indexes into an array of
  # size env.action_space.n predicted rewards.  I.e. given a state s, q[s] is
  # the predicted rewards for moving left (0) and right (1).
  q = np.zeros((
    STATE_BUCKETS,
    STATE_BUCKETS,
    STATE_BUCKETS,
    STATE_BUCKETS,
    env.action_space.n
  ))

  train(env, q)
  env.close()

  # New env with human rendering for testing the result.
  env = gym.make("CartPole-v1", render_mode="human")
  test(env, q)
  env.close()
