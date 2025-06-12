#!/bin/bash
# Use these commands to run the experiments on the respective model configurations.

# Run experiment for all noop policy in Figure 5 (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "noop" --pretrain_steps 0 --explore 0 --cuda no --seed 0

# Run experiment for random policy in Figure 5 (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "random" --pretrain_steps 0  --explore 0 --cuda no --seed 0

# For the 55k parameter model in Figure 5 with inference time equal to 1 step, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 1 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 1 --noops 1 --seed 0

# For the 1M parameter model in Figure 5 with inference time equal to 5 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 5 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 5 --noops 5 --seed 0

# For the 10M parameter model in Figure 5 with inference time equal to 8 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 8 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 8 --noops 8 --seed 0

# For the 100M parameter model in Figure 5 with inference time equal to 70 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 70 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 70 --noops 70 --seed 0

# For the 1B parameter model in Figure 5 with inference time equal to 596 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 596 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_tetris.py --model "dqn" --cuda yes --delay 596 --noops 596 --seed 0
