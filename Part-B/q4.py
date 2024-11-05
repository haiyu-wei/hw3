import matplotlib.pyplot as plt
import numpy as np

# Sample data: Replace these with your actual measurements
K_values = np.array([1, 5, 10, 50, 100])  # in millions

# Execution times in seconds (replace with your actual data)
cpu_times = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
gpu_times_no_unified = {
    '1 block, 1 thread': np.array([0.2, 1.0, 2.0, 10.0, 20.0]),
    '1 block, 256 threads': np.array([0.05, 0.25, 0.5, 2.5, 5.0]),
    'Multiple blocks, 256 threads/block': np.array([0.01, 0.05, 0.1, 0.5, 1.0])
}
gpu_times_unified = {
    '1 block, 1 thread': np.array([0.15, 0.75, 1.5, 7.5, 15.0]),
    '1 block, 256 threads': np.array([0.04, 0.2, 0.4, 2.0, 4.0]),
    'Multiple blocks, 256 threads/block': np.array([0.008, 0.04, 0.08, 0.4, 0.8])
}

# Plotting without Unified Memory
plt.figure(figsize=(10, 6))
plt.plot(K_values, cpu_times, 'o-', label='CPU Only')
for label, times in gpu_times_no_unified.items():
    plt.plot(K_values, times, 'o-', label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('K (in millions)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. K (Without Unified Memory)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("plt1.png")

# Plotting with Unified Memory
plt.figure(figsize=(10, 6))
plt.plot(K_values, cpu_times, 'o-', label='CPU Only')
for label, times in gpu_times_unified.items():
    plt.plot(K_values, times, 'o-', label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('K (in millions)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. K (With Unified Memory)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("plt2.png")