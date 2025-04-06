import re
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_benchmark(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    results = defaultdict(lambda: defaultdict(list))  # Group by k and efR
    current_batch = False
    query_count = 9000
    recall_sum = 0
    row_count = 0
    final_time = 0  # Store the time of the last query in the batch
    efR = 0
    k = 0
    for line in lines:
        batch_match = re.match(r"Running \d+ queries benchmark with params: efR: (\d+), k: (\d+)", line)
        query_match = re.match(r"\[\d+\.\d+\] Query \d+: recall=([\d\.]+), time=([\d\.]+) seconds", line)

        if batch_match:
            # If we already collected data, save the previous batch
            if current_batch:
                avg_recall = recall_sum / row_count
                avg_latency = final_time / query_count  # Using the last query's time for the batch
                avg_qps = query_count / final_time  # Queries per second (using the last query's time)
                results[k][efR] = (avg_recall, avg_qps, avg_latency)  # Save QPS and Latency

            efR = batch_match.group(1)
            k = batch_match.group(2)
            recall_sum = 0
            row_count = 0
            final_time = 0  # Reset for the new batch
            current_batch = True

        elif query_match:
            # batch_match = None
            recall = float(query_match.group(1))
            time = float(query_match.group(2))
            recall_sum += recall
            final_time = time  # Keep only the time of the last query in the batch
            row_count += 1

    # Save the last batch
    if current_batch:
        avg_recall = float(recall_sum / row_count)
        avg_latency = final_time / query_count  # Using the last query's time for the batch
        avg_qps = query_count / final_time  # Queries per second (using the last query's time)
        results[k][efR] = (avg_recall, avg_qps, avg_latency)  # Save QPS and Latency

    return results

def save_results_to_csv(results, output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["efR", "k", "Average Recall", "QPS", "Avg Latency (s)"])
        for k, batch in results.items():
            for ef_r, batch_results in batch.items():
                writer.writerow([ef_r, k, round(batch_results[0], 2), round(batch_results[1], 2), round(batch_results[2], 2)])

def plot_results(results):
    # Sort efR values numerically
    k_qps_list = {}
    k_latency_list = {}
    efR_values = []
    graph_data = {}
    for k, batch in results.items():
        graph_data[k] = {}
        graph_data[k]["efR_values"] = batch.keys()
        graph_data[k]["qps_list"] = []
        graph_data[k]["k_latency_list"] = []
        for ef_r, batch_values in batch.items():
            graph_data[k]["qps_list"].append(batch_values[1])  # QPS values
            graph_data[k]["k_latency_list"].append(batch_values[2])  # Latency values

    # Plot QPS
    plt.figure(figsize=(10, 6))
    for k in graph_data.keys():
        plt.plot(graph_data[k]["efR_values"], graph_data[k]["qps_list"], marker='o', label=f'k = {k}')
    plt.xlabel("efR")
    plt.ylabel("Queries Per Second (QPS)")
    plt.xticks(rotation=45)
    plt.title("Queries Per Second (QPS) vs efR for Different k Values")
    plt.legend()
    plt.grid()
    plt.savefig("query_graphs/qps.png")  # Save as PNG
    plt.close()

    # Plot Latency
    plt.figure(figsize=(10, 6))
    for k in graph_data.keys():
        plt.plot(graph_data[k]["efR_values"], graph_data[k]["k_latency_list"], marker='o', label=f'k = {k}')
    plt.xlabel("efR")
    plt.ylabel("Average Latency [s]")
    plt.xticks(rotation=45)
    plt.title("Average Latency [s] vs efR for Different k Values")
    plt.legend()
    plt.grid()
    plt.savefig("query_graphs/avg_latency.png")  # Save as PNG
    plt.close()

# Example usage
file_path = "results_M_60_efC_75_vec_10000000_q_10000_madvise_None_bs_10240_mem_limit_2G_pid_171927_log.txt"  # Change this to your log file
output_csv = "benchmark_query_results.csv"
results = parse_benchmark(file_path)
save_results_to_csv(results, output_csv)
plot_results(results)
