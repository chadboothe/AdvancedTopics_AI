
#------------------------------
# Step 4:  Take a User Input & Create A Factorial Calculator
#------------------------------
#

# Write Code Here

#------------------------------
# Step 5: Download the time library (even though it is pre-installed)
#------------------------------

# Write Code Here

#------------------------------
# Step 6 Code Cell (DON'T EDIT)
#------------------------------

import torch
import time

# Matrix Multiplication Function
def matrix_multiplication(size, device):
  A = torch.randn(size, size, device=device)
  B = torch.randn(size, size, device=device)
  start_time = time.time()
  C = torch.mm(A, B)
  end_time = time.time()

  return end_time - start_time

# Matrix Size
size = 10000

#CUDA (Nvidia)
if torch.cuda.is_available():
  gpu_time = matrix_multiplication(size, device="cuda")
  print(f"Matrix multiplication on GPU took {gpu_time:.6f} seconds")
#MPS (Apple Mx)
if torch.backends.mps.is_available():
  gpu_time = matrix_multiplication(size, device="mps")
  print(f"Matrix multiplication on GPU took {gpu_time:.6f} seconds")

cpu_time = matrix_multiplication(size, device="cpu")
print(f"Matrix multiplication on CPU took {cpu_time:.6f} seconds")