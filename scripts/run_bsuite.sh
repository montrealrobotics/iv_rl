for seed in 0 1 ...19
do
	python bsuite/main.py --env $1/$seed --model $2 
done