for env_seed in 0123 1123 2123 3123 4123
do
	for net_seed in 0123 1123 2123 3123 4123
	do
		python main.py --env $1 --model $2 --net_seed $net_seed --env_seed $env_seed
	done
done