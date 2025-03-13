import psutil
import time
import csv
import argparse

def collect_process_memory(pid, output_file):
    """ Continuously log memory usage of a process to a CSV file until it dies. """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time_stamp", "memory_bytes"])  # Write header

        i = 0
        print_interval = 600  # Print every 10 s
        while True:
            try:
                proc = psutil.Process(pid)
                memory = proc.memory_info().rss  # Get memory in bytes
                timestamp = time.time()  # Unix timestamp (float)

                writer.writerow([timestamp, memory])
                if i % print_interval == 0:
                    print(f"{timestamp}, rss: {memory} bytes, {memory / 1024 / 1024 / 1024} GB")  # Print for debugging

                time.sleep(1)  # Adjust sampling interval if needed
                i += 1
            except psutil.NoSuchProcess:
                print(f"Process {pid} has ended.")
                break

def generate_file_name(M, efC, num_vectors, num_queries, madvise, block_size, process_limit_high):
    return f"results_M_{M}_efC_{efC}_vec_{num_vectors}_q_{num_queries}_madvise_{madvise}_bs_{block_size}_mem_limit_{process_limit_high}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor RSS memory usage of a process.")
    parser.add_argument("pid", type=int, help="PID of the process to monitor")
    args = parser.parse_args()

    pid = args.pid  # Get the PID from the command line argument
    run_name = generate_file_name(M=60, efC=75, num_vectors=10_000_000, num_queries=10_000, madvise="None", block_size=10_240, process_limit_high="55G")
    output_file = f"{run_name}_pid_{pid}_rss_memory_monitor.csv"
    print("Start collecting memory usage for process", pid)
    collect_process_memory(pid, output_file)
