# ============================================================================
# MAIN SCRIPT
# ============================================================================

import time
import sys
sys.path.insert(0, '..')
from bitalino import BITalino
from processor import EMGProcessor

# The macAddress variable on Windows can be "XX:XX:XX:XX:XX:XX" or "COMX"
# while on Mac OS can be "/dev/tty.BITalino-XX-XX-DevB" for devices ending with the last 4 digits of the MAC address or "/dev/tty.BITalino-DevB" for the remaining
macAddress = "COM3"

# This example will collect data for 5 sec.
running_time = 5

batteryThreshold = 30
acqChannels = [0, 1, 2, 3, 4, 5]
samplingRate = 1000
nSamples = 10
digitalOutput_on = [1, 1]
digitalOutput_off = [0, 0]

# Initialize EMG processor
emg_processor = EMGProcessor(sampling_rate=samplingRate, num_channels=len(acqChannels))

# Connect to BITalino
device = BITalino(macAddress)

# Set battery threshold
device.battery(batteryThreshold)

# Read BITalino version
print("Device version:", device.version())
print("\n" + "="*60)
print("Starting EMG acquisition and processing...")
print("="*60 + "\n")

# Start Acquisition
device.start(samplingRate, acqChannels)

start = time.time()
end = time.time()
iteration = 0

while (end - start) < running_time:
    # Read samples
    raw_data = device.read(nSamples)
    
    # Process data
    processed_data = emg_processor.process_sample(raw_data)
    
    # Display metrics every 10 iterations (~0.1 seconds)
    if iteration % 10 == 0:
        print(f"\nTime: {end - start:.2f}s")
        print("-" * 60)
        
        for ch_idx, ch_data in processed_data.items():
            if ch_data['features'] is not None:
                features = ch_data['features']
                print(f"Channel {acqChannels[ch_idx]}:")
                print(f"  RMS: {features['rms']:.2f} mV")
                print(f"  MAV: {features['mav']:.2f} mV")
                print(f"  Max Amplitude: {features['max_amplitude']:.2f} mV")
                print(f"  Zero Crossings: {features['zero_crossings']}")
                print(f"  Waveform Length: {features['waveform_length']:.2f}")
        
        # Example: Detect muscle activation based on RMS threshold
        threshold_rms = 50  # mV, adjust based on your application
        for ch_idx, ch_data in processed_data.items():
            if ch_data['features'] is not None:
                if ch_data['features']['rms'] > threshold_rms:
                    print(f"  >>> ACTIVATION DETECTED on Channel {acqChannels[ch_idx]}!")
    
    iteration += 1
    end = time.time()

print("\n" + "="*60)
print("Acquisition complete!")
print("="*60 + "\n")

# Turn BITalino led and buzzer on
device.trigger(digitalOutput_on)

# Script sleeps for n seconds
time.sleep(1)

# Turn BITalino led and buzzer off
device.trigger(digitalOutput_off)

# Stop acquisition
device.stop()

# Close connection
device.close()

print("Connection closed. Processing complete.")