import re
import numpy as np
import matplotlib.pyplot as plt
import csv

def format_index_size(size):
    if size < 1_000_000:
        return f"{size // 1_000}K"
    else:
        return f"{size / 1_000_000:.1f}M"

def parse_index_log(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    index_data = []

    time_pattern = re.compile(r'Building (\d+) vectors time:  T([\d.]+) seconds')
    memory_pattern = re.compile(r'Current index memory usage: .* ([\d.]+) GB')

    prev_time = None

    for i in range(len(lines)):
        time_match = time_pattern.search(lines[i])
        if time_match:
            index_size = int(time_match.group(1))
            time_elapsed = float(time_match.group(2))

            memory_match = memory_pattern.search(lines[i+1])  # Memory info is on the next line
            if memory_match:
                memory_usage = float(memory_match.group(1))

                # Compute time difference in hours
                batch_time_hr = 0 if prev_time is None else (time_elapsed - prev_time) / 3600
                prev_time = time_elapsed

                # Format index size
                index_size_formatted = format_index_size(index_size)

                index_data.append((index_size_formatted, round(batch_time_hr, 2), f"{memory_usage:.2f}"))

    # Print the result
    print(f"{'Index Size':<12} {'Batch Time (hr)':<15} {'Memory Usage (GB)':<18}")
    print("=" * 50)
    for row in index_data:
        print(f"{row[0]:<12} {row[1]:<15} {row[2]:<18}")

    # Export data to CSV
    csv_filename = 'index_data.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index Size", "Batch Time (hr)", "Memory Usage (GB)"])  # Write header
        writer.writerows(index_data)  # Write the data

    print(f"Data saved to {csv_filename}")

    return index_data

# Run the script with your log file
index_data = parse_index_log("results_M_60_efC_75_vec_10000000_q_10000_madvise_None_bs_10240_mem_limit_2G_pid_171927_log.txt")


# Extracting data for the graph
index_sizes = [row[0] for row in index_data]  # Index sizes (formatted as K/M)
batch_times = [row[1] for row in index_data]  # Batch elapsed times
memory_usages = [round(float(row[2]), 2) for row in index_data]  # Memory usage

# Convert index sizes to numerical values and scale to millions
index_sizes_numeric = []
for size in index_sizes:
    if size.endswith("K"):
        index_sizes_numeric.append(float(size[:-1]) * 1e3 / 1e6)  # Convert to millions
    elif size.endswith("M"):
        index_sizes_numeric.append(float(size[:-1]))  # Already in millions
    else:
        index_sizes_numeric.append(float(size) / 1e6)  # Convert to millions

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Batch Time on the left y-axis
ax1.set_xlabel("Index Size (vectors / 1M)")
ax1.set_ylabel("Batch Time (hr)", color="tab:blue")
ax1.plot(index_sizes_numeric, batch_times, marker='o', linestyle='-', color="tab:blue", label="Batch Time (hr)")
ax1.tick_params(axis='y', labelcolor="tab:blue")

# Set the range and ticks for the left y-axis (Batch Time)
ax1.set_ylim(0, 100)  # Adjust this based on your data range
ax1.set_yticks(np.arange(0, 101, 10))  # Set tick marks every 10 units (you can adjust this)

# Create a second y-axis to plot Memory Usage
ax2 = ax1.twinx()
ax2.set_ylabel("Memory Usage (GB)", color="tab:green")
ax2.plot(index_sizes_numeric, memory_usages, marker='s', linestyle='--', color="tab:green", label="Memory Usage (GB)")
ax2.tick_params(axis='y', labelcolor="tab:green")


# Set the range and ticks for the right y-axis (Memory Usage)
ax2.set_ylim(0, 3)  # Adjust this based on your data range
ax2.set_yticks(np.arange(0, 3.1, 0.5))  # Set tick marks every 0.5 units (you can adjust this)


# Title and grid
plt.title("Build Index")
ax1.grid(True)

x_value_5M = 5  # Since x-axis is in millions

# Add a vertical dashed line at x = 5M
ax1.axvline(x=x_value_5M, color='red', linestyle='--', linewidth=1)

# Add a note next to the line
y_pos = 0.9
ax1.text(
    x_value_5M + 1.5,  # Slightly shift the text to the right
    ax1.get_ylim()[1] * y_pos,  # Position at 80% of the y-axis max value
    "Process limit: 5GB",
    fontsize=12, color="red",
    ha="left", va="center",
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)

ax1.text(
    x_value_5M - 1.5,  # Slightly shift the text to the right
    ax1.get_ylim()[1] * y_pos,  # Position at 80% of the y-axis max value
    "Process limit: 2GB",
    fontsize=12, color="red",
    ha="right", va="center",
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
)


# Save the figure to a file
plt.tight_layout()  # Adjust layout to make sure everything fits
plt.savefig("index_growth.png", dpi=300, bbox_inches="tight")

print("Graph saved as 'index_growth.png'")
