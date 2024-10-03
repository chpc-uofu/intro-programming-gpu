# To run:
module load gcc/13.1.0
module load cuda/12.4.0

# 2:: Using mutiple BLOCK where each BLOCK contains 16*16 THREADS
# Target: NVIDIA V100 (sm_70) architecture
make clean
make
./mul2
