# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple agent-environment training loop."""

from bsuite.baselines import base
from bsuite.logging import terminal_logging

import dm_env

import numpy as np
import wandb

def run(agent: base.Agent,
        train_environment: dm_env.Environment,
        test_environment: dm_env.Environment,
        num_episodes: int,
        verbose: bool = False) -> None:
  """Runs an agent on an environment.

  Note that for bsuite environments, logging is handled internally.

  Args:
    agent: The agent to train and evaluate.
    environment: The environment to train on.
    num_episodes: Number of episodes to train for.
    verbose: Whether to also log to terminal.
  """

  if verbose:
    test_environment = terminal_logging.wrap_environment(
        test_environment, log_every=True)  # pytype: disable=wrong-arg-types

  train_scores, test_scores = [], []
  for i_episode in range(num_episodes):
    # Run an episode.
    score = 0
    ep_var, ep_weights, eff_bs_list, eps_list = [], [], [], []
    timestep = train_environment.reset()
    while not timestep.last():
      # Generate an action from the agent's policy.
      action = agent.select_action(timestep)

      # Step the environment.
      new_timestep = train_environment.step(action)

      # Tell the agent about what just happened.
      logs = agent.update(timestep, action, new_timestep)

      if len(logs) > 0:
        ep_var.extend(logs[0])
        ep_weights.extend(logs[1])
        eff_bs_list.append(logs[2])
        eps_list.append(logs[3])


      # Book-keeping.
      timestep = new_timestep

      score += timestep.reward
    train_scores.append(score)
    if i_episode % 1 == 0:
      test_score = test(agent, test_environment)
      test_scores.append(test_score)
      # wandb.log({"Test Return": test_score, "Test Return / 100 episodes": np.mean(test_scores[-100:])}, commit=False)
    # if len(ep_var) > 0:
    #   agent.train_log(ep_var, ep_weights, eff_bs_list, eps_list)
    # wandb.log({"Train Return": score, "Train Return / 100 episodes": np.mean(train_scores[-100:])})
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(train_scores[-100:])), end="")




def test(agent, environment):
  score = 0
  timestep = environment.reset()
  while not timestep.last():
    action = agent.select_action_test(timestep)
    new_timestep = environment.step(action)
    timestep = new_timestep
    score += timestep.reward

  return score