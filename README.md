# IV_RL

## Installing Dependencies (Python version: 3.8.8)

pip install -r requirements.txt

## Adding Python Paths for Mujoco

	export PYTHONPATH=$PYTHONPATH:$(pwd)/rlkit/
	export PYTHONPATH=$PYTHONPATH:$(pwd)/mbbl_envs/


## Running Mujoco Experiments 

	sh ./scripts/run_mujoco.sh <env_name> <model_name>


env_name can be ["gym_walker2d", "gym_cheetah", "gym_ant", "gym_hopper"]

model_name can be ["SAC", "ProbSAC", "EnsembleSAC", "PredEnsembleSAC", "IV_EnsembleSAC", "IV_ProbEnsembleSAC", "IV_SAC"]


For eg:

	sh ./scripts/run_mujoco.sh gym_walker2d IV_SAC

## Running Bsuite Experiments

	sh ./scripts/run_bsuite.sh cartpole_noise <model_name> 

model_name can be ["DQN", "BootstrapDQN", "ProbEnsembleDQN", "IV_DQN", "IV_ProbEnsembleDQN"]

For eg:
	
	sh ./scripts/run_bsuite.sh cartpole_noise IV_DQN 


## Running Gym Experiments

	sh ./scripts/run_gym.sh <env_name> <model_name>

env_name can be ["LunarLander-v2" or "MountainCar-v0"]

model_name can bet ["DQN", "EnsembleDQN", "ProbDQN", "IV_ProbDQN", "BootstrapDQN", "IV_ProbEnsembleDQN", "IV_DQN", "IV_BootstrapDQN"]
