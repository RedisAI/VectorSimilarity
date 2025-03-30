import pandas as pd
import matplotlib.pyplot as plt
import re

# ---- 1. Process CSV File (Memory Usage) ----
def plot_memory_usage(csv_file, output_file):
    df = pd.read_csv(csv_file)
    # Convert memory from bytes to GB
    df["memory_gb"] = df["memory_bytes"] / (1024**3)

    num_entries = 1000
    step = 5
    df = df.head(num_entries).iloc[::step]

    plt.figure(figsize=(30, 20))
    plt.scatter(df["time_stamp"], df["memory_gb"], label="Memory Usage (GB)", color='b', marker='o', s=50)
    plt.xlabel("Time (Unix Timestamp)")
    plt.ylabel("Memory (GB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.grid()

    plt.savefig(output_file)  # Save as PNG file
    plt.close()  # Close the plot to free memory

# ---- 2. Process Log File (Index Size) ----
def parse_log_file(log_file):
    pattern = re.compile(r"\[(\d+\.\d+)\] Building (\d+) vectors time:  T(\d+\.\d+) seconds")

    timestamps = []
    index_sizes_m = []

    with open(log_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp, index_size, _ = match.groups()
                timestamps.append(float(timestamp))
                index_sizes_m.append(int(index_size) / 1_000_000)  # Convert to millions

    return timestamps, index_sizes_m

def plot_index_size(log_file, output_file):
    timestamps, index_sizes = parse_log_file(log_file)

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, index_sizes, label="Index Size (vectors)", color='r')
    plt.xlabel("Time (Unix Timestamp)")
    plt.ylabel("Index Size (vectors)")
    plt.title("Index Size Over Time")
    plt.legend()
    plt.grid()

    plt.savefig(output_file)  # Save as PNG file
    plt.close()  # Close the plot to free memory

def parse_log_file(log_file):
    pattern = re.compile(r"\[(\d+\.\d+)\] Building (\d+) vectors time:  T(\d+\.\d+) seconds")
    timestamps = []
    index_sizes_m = []  # Store index sizes in millions

    with open(log_file, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp, index_size, _ = match.groups()
                timestamps.append(float(timestamp))
                index_sizes_m.append(int(index_size) / 1_000_000)  # Convert to millions

    return timestamps, index_sizes_m

def plot_combined(csv_file, log_file, output_file, num_entries=None, step=5):
    # Read the CSV file for memory usage
    df = pd.read_csv(csv_file)

    if num_entries is None:
        num_entries = len(df)

    # Convert memory from bytes to GB
    df["memory_gb"] = df["memory_bytes"] / (1024**3)

    # Subtract the first memory value to adjust all memory values
    initial_memory = df["memory_gb"].iloc[0]
    df["memory_gb"] = df["memory_gb"] - initial_memory


    # Select only the first `num_entries` rows and take every `step`-th row
    df = df.head(num_entries).iloc[::step]

    # Parse the log file for index size data
    timestamps, index_sizes_m = parse_log_file(log_file)

    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # Plot memory usage on the first y-axis (left)
    ax1.set_xlabel("Time (Unix Timestamp)")
    ax1.set_ylabel("Memory Usage (GB)", color='b')
    ax1.scatter(df["time_stamp"], df["memory_gb"], color='b', marker='o', s=10, label="Memory Usage")
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for the index size
    ax2 = ax1.twinx()
    ax2.set_ylabel("Index Size (M vectors)", color='r')
    ax2.scatter(timestamps, index_sizes_m, color='r', marker='x', s=10, label="Index Size")
    ax2.tick_params(axis='y', labelcolor='r')

    # Set the title and legend
    plt.title("Memory Usage and Index Size Over Time")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.grid(True)

    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()  # Close the plot to free memory

    print(f"Combined plot saved to {output_file}")

# ---- Run the plots ----
result_details = "results_M_60_efC_75_vec_10000000_q_10000_madvise_MADV_DONTNEED_bs_10240"
csv_file = f"results/mem_monitor/{result_details}_pid_1335459_rss_memory_monitor.csv"
log_file = f"results/logs/{result_details}_log.txt"

# plot_memory_usage(csv_file, f"results/graphs/{result_details}_memory_usage.png")
# plot_index_size(log_file, f"results/graphs/{result_details}_index_size.png")
plot_combined(csv_file, log_file, f"results/graphs/{result_details}_combined.png", step = 20)

print("Graphs saved: memory_usage.png, index_size.png")
