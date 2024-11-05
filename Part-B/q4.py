import matplotlib.pyplot as plt
import numpy as np

# K values (in millions)
K_values = np.array([1, 5, 10, 50, 100])

# Step 2: Without Unified Memory - Execution times for different scenarios
step2_times = {
    '1 block, 1 thread': np.array([8.6697e-05, 8.5092e-05, 8.0092e-05, 7.6767e-05, 5.3616e-05]),
    '1 block, 256 threads': np.array([8.5403e-05, 9.1455e-05, 8.1256e-05, 7.7559e-05, 5.2096e-05]),
    'Multiple blocks, 256 threads/block': np.array([1.00696e-04, 1.60425e-04, 2.23158e-04, 3.8399e-04, 1.49588e-03])
}

# Step 3: With Unified Memory - Execution times for different scenarios
step3_times = {
    '1 block, 1 thread': np.array([6.15171e-04, 4.32272e-04, 4.55753e-04, 4.52795e-04, 7.83571e-04]),
    '1 block, 256 threads': np.array([3.38289e-04, 4.29447e-04, 4.41302e-04, 4.74693e-04, 8.57625e-04]),
    'Multiple blocks, 256 threads/block': np.array([2.90941e-03, 1.2437e-02, 2.43197e-02, 1.17058e-01, 2.37736e-01])
}

# Plotting Step 2 - Without Unified Memory
plt.figure(figsize=(10, 6))
for label, times in step2_times.items():
    plt.plot(K_values, times, 'o-', label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('K (in millions)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. K (Without Unified Memory)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("execution_time_step2_no_unified.png")
plt.show()

# Plotting Step 3 - With Unified Memory
plt.figure(figsize=(10, 6))
for label, times in step3_times.items():
    plt.plot(K_values, times, 'o-', label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('K (in millions)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. K (With Unified Memory)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("execution_time_step3_unified.png")
plt.show()