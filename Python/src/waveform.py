import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_name = r'Python\src\Datasets\Satellite_Data\Whistler Wave Database\20th January 2016\Waveform_20160120_t19_0_04.csv'
df = pd.read_csv(file_name)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df['time (s)'], df['B_wave'], label='B_wave', color='blue', linewidth=0.8)

# Formatting the chart
plt.title('Time-domain: Waveform Amplitude Over Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Display the plot
plt.tight_layout()
plt.savefig("time_domain_plot")