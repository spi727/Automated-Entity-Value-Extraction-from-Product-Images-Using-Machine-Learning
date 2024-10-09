import cupy as cp

# Check if CuPy can access the GPU
is_cuda_available = cp.cuda.is_available()
if is_cuda_available:
    device_count = cp.cuda.runtime.getDeviceCount()
    print("CUDA is available.")
    print("Number of GPUs:", device_count)
else:
    print("CUDA is not available.")
