import time
import numpy as np
from scipy import signal
from collections import deque
from bitalino import BITalino

# ============================================================================
# EMG PROCESSING CLASS
# ============================================================================
class EMGProcessor:
    """
    EMG Signal Processing Pipeline optimized for benchmark EMG databases.
    
    Based on standard preprocessing for EMG classification datasets with:
    - 10 electrode channels
    - Sampling rate: 2000 Hz
    - EMG frequency content: 20-500 Hz
    - 4th order Butterworth bandpass filter
    - Notch filter for 50/60 Hz powerline interference
    """
    
    def __init__(self, sampling_rate=2000, num_channels=10, powerline_freq=50, input_in_volts=False):
        self.fs = sampling_rate
        self.num_channels = num_channels
        self.powerline_freq = powerline_freq
        self.input_in_volts = input_in_volts  # NEW: flag for raw data units
        
        # Design bandpass filter (20-500 Hz for EMG signal content)
        # These frequencies encompass the typical EMG spectrum
        nyquist = self.fs / 2.0
        low_freq = 20 / nyquist
        high_freq = 500 / nyquist
        
        # Clamp frequencies to valid range (0.001 to 0.999)
        low_freq = max(0.001, min(low_freq, 0.999))
        high_freq = max(low_freq + 0.001, min(high_freq, 0.999))
        
        # 4th order Butterworth filter - good balance between roll-off and phase response
        self.sos_bp = signal.butter(4, [low_freq, high_freq], btype='band', output='sos')
        
        # Design notch filter for powerline interference (50 Hz or 60 Hz)
        notch_freq = self.powerline_freq / nyquist
        notch_freq = max(0.001, min(notch_freq, 0.999))
        # High Q factor (Q=30) for narrow notch around powerline frequency
        self.b_notch, self.a_notch = signal.iirnotch(notch_freq, 30)
        # Convert notch filter to SOS format for numerical stability
        self.sos_notch = signal.tf2sos(self.b_notch, self.a_notch)
        
        # Initialize filter states for each channel
        self.zi_bp = [signal.sosfilt_zi(self.sos_bp) for _ in range(num_channels)]
        self.zi_notch = [signal.sosfilt_zi(self.sos_notch) for _ in range(num_channels)]
        
        # Buffer for storing recent data (for computing metrics over windows)
        # 500ms buffer for feature computation
        self.buffer_size = int(0.5 * self.fs)
        self.data_buffer = [deque(maxlen=self.buffer_size) for _ in range(num_channels)]
        
    def convert_to_voltage(self, raw_data):
        """
        Convert ADC values to voltage in mV only if input is not already in volts.
        
        Assumes 12-bit ADC (0-4095) with 5V reference (common for EMG devices).
        This matches typical benchmark database specifications.
        """
        if self.input_in_volts:
            return raw_data  # skip conversion
        # Standard 12-bit ADC with 5V reference
        ADC_BITS = 12
        ADC_VREF = 5.0  # 5V reference voltage
        
        # Convert ADC counts to voltage (mV)
        adc_max = (2 ** ADC_BITS) - 1
        voltage = (raw_data / adc_max) * ADC_VREF * 1000
        
        return voltage
    
    def apply_filters(self, channel_data, channel_idx):
        """Apply bandpass and notch filters with state preservation"""
        # Bandpass filter using second-order sections (more numerically stable)
        filtered_data, self.zi_bp[channel_idx] = signal.sosfilt(
            self.sos_bp, channel_data, zi=self.zi_bp[channel_idx]
        )
        
        # Notch filter
        filtered_data, self.zi_notch[channel_idx] = signal.sosfilt(
            self.sos_notch, filtered_data, zi=self.zi_notch[channel_idx]
        )
        
        return filtered_data
    
    def rectify(self, data):
        """Full-wave rectification"""
        return np.abs(data)
    
    def envelope(self, data, window_ms=50):
        """
        Calculate moving average envelope (Low-pass filter on rectified signal).
        
        Standard envelope extraction for EMG with 50ms window (typical for movement analysis).
        At 2000 Hz sampling: 50ms = 100 samples
        """
        window_size = max(1, int(window_ms * self.fs / 1000))
        # Use hann window for smoother envelope
        window = np.hanning(window_size)
        window = window / window.sum()
        return np.convolve(data, window, mode='same')
    
    def compute_rms(self, data):
        """Root Mean Square"""
        return np.sqrt(np.mean(data**2))
    
    def compute_mav(self, data):
        """Mean Absolute Value"""
        return np.mean(np.abs(data))
    
    def compute_zero_crossings(self, data, threshold=0):
        """Count zero crossings"""
        crossings = np.where(np.diff(np.sign(data - threshold)))[0]
        return len(crossings)
    
    def compute_waveform_length(self, data):
        """Waveform Length - cumulative change in signal"""
        return np.sum(np.abs(np.diff(data)))
    
    def compute_features(self, data):
        """Compute all EMG features"""
        features = {
            'rms': self.compute_rms(data),
            'mav': self.compute_mav(data),
            'zero_crossings': self.compute_zero_crossings(data),
            'waveform_length': self.compute_waveform_length(data),
            'max_amplitude': np.max(np.abs(data)),
            'variance': np.var(data)
        }
        return features
    
    def process_sample(self, raw_sample):
        """Process a single sample batch from BITalino"""
        # Extract analog channels (columns 5 onwards)
        analog_data = raw_sample[:, 5:]
        
        processed_channels = {}
        
        for ch_idx in range(min(analog_data.shape[1], self.num_channels)):
            # Convert to voltage
            voltage_data = self.convert_to_voltage(analog_data[:, ch_idx])
            
            # Apply filters
            filtered_data = self.apply_filters(voltage_data, ch_idx)
            
            # Rectify
            rectified_data = self.rectify(filtered_data)
            
            # Calculate envelope
            envelope_data = self.envelope(rectified_data)
            
            # Update buffer
            self.data_buffer[ch_idx].extend(filtered_data)
            
            # Compute features if buffer is sufficiently filled
            if len(self.data_buffer[ch_idx]) >= 100:
                buffer_array = np.array(self.data_buffer[ch_idx])
                features = self.compute_features(buffer_array)
            else:
                features = None
            
            processed_channels[ch_idx] = {
                'raw': voltage_data,
                'filtered': filtered_data,
                'rectified': rectified_data,
                'envelope': envelope_data,
                'features': features
            }
        
        return processed_channels