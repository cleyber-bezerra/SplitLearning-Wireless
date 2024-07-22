import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def process_file(file_path, color, label):
    df = pd.read_csv(file_path)

    # Calculate mean and standard error for rx_packets grouped by client
    grouped = df.groupby('Flow ID')['Throughput (Mbps)'].agg(['mean', 'std', 'count'])
    grouped['stderr'] = grouped['std'] / np.sqrt(grouped['count'])

    # Calculate confidence interval (95% CI)
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    grouped['ci_low'] = grouped['mean'] - z_score * grouped['stderr']
    grouped['ci_high'] = grouped['mean'] + z_score * grouped['stderr']

    # Prepare data for plotting
    clients = grouped.index
    mean_rx_packets = grouped['mean']
    ci_low = grouped['ci_low']
    ci_high = grouped['ci_high']

    # Ensure clients include all values from 1 to 20
    clients_complete = np.arange(1, 6)

    # Plot mean rx_packets
    plt.plot(clients_complete, mean_rx_packets, marker='', linestyle='-', linewidth=1, color=color, label=label)

    # Shade the area between confidence intervals
    plt.fill_between(clients_complete, ci_low, ci_high, color=color, alpha=0.1, linewidth=0)

# File paths
files = ['low_loss.csv', 'medium_loss.csv', 'high_loss.csv']
colors = ['g', 'b', 'r']
labels = ['Low Loss', 'Moderate Loss', 'High Loss']

# Plotting setup
plt.figure(figsize=(10, 6))

for file, color, label in zip(files, colors, labels):
    process_file(file, color, label)

plt.title('Average throughput for each client')
plt.xlabel('Clients')
plt.ylabel('Throughput (Mbps)')
plt.grid(True)
plt.legend(loc='upper right')
plt.xticks(np.arange(1, 6), rotation=0)
plt.ylim(0, None)  # Force y-axis to start from zero
plt.tight_layout()

# Save and show the plot
name_output_file = "packets_low_moderate_high_losses"
plt.savefig(name_output_file + ".png")
plt.show()

