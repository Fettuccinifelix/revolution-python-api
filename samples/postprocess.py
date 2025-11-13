import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# ============================================================================ #
# CONFIGURATION
# ============================================================================ #

# Choose whether the CSV data is already in volts
data_in_volts = True  # True if CSV already contains voltage data, False if ADC counts

# Sampling rate and other EMG parameters
sampling_rate = 2000  # Hz, typical for EMG classification datasets

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Add parent directory to path for imports
sys.path.insert(0, parent_dir)
from processor import EMGProcessor

# ============================================================================ #
# LOAD CSV DATA
# ============================================================================ #

csv_file = os.path.join(parent_dir, 'output.csv')
print(f"Loading data from {csv_file}...")
data = pd.read_csv(csv_file, header=None)
print(f"Data shape: {data.shape}")
print(f"Number of channels: {data.shape[1]}")
print(f"Number of samples: {data.shape[0]}")

num_channels = data.shape[1]

# Initialize EMGProcessor
print("\nInitializing EMGProcessor...")
processor = EMGProcessor(sampling_rate=sampling_rate, num_channels=num_channels, input_in_volts=data_in_volts)

# ============================================================================ #
# PROCESS ALL CHANNELS
# ============================================================================ #

print("Processing all channels through the pipeline...")
all_data = data.values
processed_data = {}

for ch_idx in range(num_channels):
    print(f"\nProcessing Channel {ch_idx}...")
    raw_channel_data = all_data[:, ch_idx]

    # Convert to voltage only if data is ADC counts
    if not data_in_volts:
        voltage_data = processor.convert_to_voltage(raw_channel_data)
    else:
        voltage_data = raw_channel_data.copy()

    # Apply filters
    filtered_data = processor.apply_filters(voltage_data, ch_idx)
    rectified_data = processor.rectify(filtered_data)
    envelope_data = processor.envelope(rectified_data)
    features = processor.compute_features(filtered_data)

    processed_data[ch_idx] = {
        'raw': raw_channel_data,
        'voltage': voltage_data,
        'filtered': filtered_data,
        'rectified': rectified_data,
        'envelope': envelope_data,
        'features': features
    }

    # Unit string for printing
    unit_str = 'mV' if not data_in_volts else 'mV'  # adjust if your CSV is in µV
    print(f"  Features for Channel {ch_idx}:")
    print(f"    RMS: {features['rms']:.4f} {unit_str}")
    print(f"    MAV: {features['mav']:.4f} {unit_str}")
    print(f"    Max Amplitude: {features['max_amplitude']:.4f} {unit_str}")
    print(f"    Zero Crossings: {features['zero_crossings']}")
    print(f"    Waveform Length: {features['waveform_length']:.4f}")
    print(f"    Variance: {features['variance']:.4f}")

# Create time axis (seconds)
time_axis = np.arange(len(all_data)) / sampling_rate

# ============================================================================ #
# PLOTTING
# ============================================================================ #

print("\nGenerating plots...")

raw_label = 'Raw Voltage (mV)' if data_in_volts else 'Raw ADC Value'
voltage_label = 'Voltage (mV)'  # always in mV after conversion

# Plot each channel's pipeline
for ch_idx in range(num_channels):
    # If CSV already contains voltage values, skip the raw ADC subplot
    if data_in_volts:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle(f'EMG Processing Pipeline - Channel {ch_idx}', fontsize=16, fontweight='bold')

        # Step 1: Voltage (input is already in volts)
        axes[0].plot(time_axis, processed_data[ch_idx]['voltage'], 'g-', linewidth=0.5)
        axes[0].set_ylabel(voltage_label, fontsize=10)
        axes[0].set_title('Step 1: Voltage (input in volts)', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Step 2: Filtered Data
        axes[1].plot(time_axis, processed_data[ch_idx]['filtered'], 'r-', linewidth=0.5)
        axes[1].set_ylabel('Filtered Voltage (mV)', fontsize=10)
        axes[1].set_title('Step 2: Bandpass (20-500 Hz) & Notch Filtered', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Step 3: Rectified Data
        axes[2].plot(time_axis, processed_data[ch_idx]['rectified'], 'orange', linewidth=0.5)
        axes[2].set_ylabel('Rectified Voltage (mV)', fontsize=10)
        axes[2].set_title('Step 3: Full-Wave Rectification', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        # Step 4: Envelope
        axes[3].plot(time_axis, processed_data[ch_idx]['envelope'], 'purple', linewidth=1)
        axes[3].set_ylabel('Envelope (mV)', fontsize=10)
        axes[3].set_xlabel('Time (s)', fontsize=10)
        axes[3].set_title('Step 4: Moving Average Envelope', fontsize=12)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'channel_{ch_idx}_pipeline.png', dpi=100, bbox_inches='tight')
        print(f"  Saved: channel_{ch_idx}_pipeline.png")
        plt.close()
    else:
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        fig.suptitle(f'EMG Processing Pipeline - Channel {ch_idx}', fontsize=16, fontweight='bold')

        # Step 1: Raw Data
        axes[0].plot(time_axis, processed_data[ch_idx]['raw'], 'b-', linewidth=0.5)
        axes[0].set_ylabel(raw_label, fontsize=10)
        axes[0].set_title('Step 1: Raw Data', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Step 2: Voltage Converted
        axes[1].plot(time_axis, processed_data[ch_idx]['voltage'], 'g-', linewidth=0.5)
        axes[1].set_ylabel(voltage_label, fontsize=10)
        axes[1].set_title('Step 2: Voltage Conversion', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Step 3: Filtered Data
        axes[2].plot(time_axis, processed_data[ch_idx]['filtered'], 'r-', linewidth=0.5)
        axes[2].set_ylabel('Filtered Voltage (mV)', fontsize=10)
        axes[2].set_title('Step 3: Bandpass (20-500 Hz) & Notch Filtered', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        # Step 4: Rectified Data
        axes[3].plot(time_axis, processed_data[ch_idx]['rectified'], 'orange', linewidth=0.5)
        axes[3].set_ylabel('Rectified Voltage (mV)', fontsize=10)
        axes[3].set_title('Step 4: Full-Wave Rectification', fontsize=12)
        axes[3].grid(True, alpha=0.3)

        # Step 5: Envelope
        axes[4].plot(time_axis, processed_data[ch_idx]['envelope'], 'purple', linewidth=1)
        axes[4].set_ylabel('Envelope (mV)', fontsize=10)
        axes[4].set_xlabel('Time (s)', fontsize=10)
        axes[4].set_title('Step 5: Moving Average Envelope', fontsize=12)
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'channel_{ch_idx}_pipeline.png', dpi=100, bbox_inches='tight')
        print(f"  Saved: channel_{ch_idx}_pipeline.png")
        plt.close()

# Summary plot: all envelopes
fig, axes = plt.subplots(num_channels, 1, figsize=(14, 12))
fig.suptitle('EMG Envelopes - All Channels Comparison', fontsize=16, fontweight='bold')

for ch_idx in range(num_channels):
    axes[ch_idx].plot(time_axis, processed_data[ch_idx]['envelope'], 'b-', linewidth=0.8)
    axes[ch_idx].set_ylabel(f'Ch {ch_idx}', fontsize=9)
    axes[ch_idx].grid(True, alpha=0.3)

    # Add feature annotations
    features = processed_data[ch_idx]['features']
    axes[ch_idx].text(
        0.98, 0.95, f"RMS: {features['rms']:.2f} | MAV: {features['mav']:.2f}",
        transform=axes[ch_idx].transAxes, fontsize=8,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

axes[-1].set_xlabel('Time (s)', fontsize=10)
plt.tight_layout()
plt.savefig('all_channels_envelopes.png', dpi=100, bbox_inches='tight')
print(f"  Saved: all_channels_envelopes.png")
plt.close()

print("\nProcessing complete!")
print(f"Generated {num_channels} individual pipeline plots and 1 summary plot.")
