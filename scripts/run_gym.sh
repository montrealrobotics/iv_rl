for env_seed in 0 1 2 3 4
do
	for net_seed in 0 1 2 3 4
	do
		python main.py --env $1 --model $2 --net_seed $net_seed --env_seed $env_seed
	done
done