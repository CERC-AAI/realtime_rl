# This script contains commands to run experiments for different model sizes to reproduce the configurations presented
# in the paper. Each experiment setup is tailored to a specific model size with the corresponding number of inference
# and learn processes.
# Use these commands to run the experiments on the respective model configurations.

# For k =1 i.e., 1M model size (both battles and catching setting in Pokemon Blue) - Figure 9 in the paper (our experiments included seed = 0, 1, and 2)
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 1 --game "blue" --setting "battles" --num_inf 7  --num_learn 1 --t_theta_max 0.04 --seed 0
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 1 --game "blue" --setting "catching" --num_inf 7  --num_learn 1 --t_theta_max 0.04 --seed 0

# For k =7 i.e., 10M model size (both battles and catching setting in Pokemon Blue) - Figure 10 in the paper (our experiments included seed = 0, 1, and 2)

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 7 --game "blue" --setting "battles" --num_inf 12  --num_learn 3 --t_theta_max 0.1 --seed 0
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 7 --game "blue" --setting "battles" --num_inf 12  --num_learn 3 --t_theta_max 0.1 --seed 0

# For k =29 i.e., 100M model size (both battles and catching setting in Pokemon Blue) - Figure 5 in the paper (our experiments included seed = 0, 1, and 2)

PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 29 --game "blue" --setting "battles" --num_inf 72  --num_learn 33 --seed 0
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 29 --game "blue" --setting "catching" --num_inf 72  --num_learn 33 --seed 0

# For running experiments to use random actions instead of noops we need to add an additional flag as shown below (our experiments included seed = 0, 1, and 2):
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_realtime_pokemon.py --k 29 --game "blue" --setting "battles" --num_inf 72  --num_learn 33 --random_not_noop --seed 0
