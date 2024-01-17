import numpy as np
import can
import time

sampling_rate = 44100   # Sampling rate of the sine wave in samples per second 
amplitude = 1       # Amplitude of the sine wave

channels = []
for freq_scale in np.arange(1.0,1 + 0.2*8, 0.2):
    frequency = 10*freq_scale     # Frequency of the sine wave in Hz 
    duration = 2        # Duration of the sine wave in seconds 
    # Generate a time vector 
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False) 
    # Generate the sine wave 
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # Scale to 255
    positive_sine_wave = [int(i) for i in (sine_wave + 1.0)*255/2]
    channels.append(positive_sine_wave)

# Add labelled data
# channels.append([0]*len(positive_sine_wave))

# train_data = np.array(channels).T
# import pickle
# with open('sine_wave_8_signals.pkl', 'wb') as b:
#     pickle.dump(train_data.tolist(), b)

# with open('val.pkl', 'wb') as b:
#     pickle.dump(train_data[:1000].tolist(), b)

channels_sine_wave = np.array(channels).T
with can.interface.Bus(interface='pcan', channel='PCAN_USBBUS1', bitrate=500000) as bus:
    for data in channels_sine_wave:
        msg = can.Message(
            arbitration_id=0x1,
            data=list(data),
            is_extended_id=False
        )
        try:
            bus.send(msg)
            print(f"Message sent on {bus.channel_info}")
        except can.CanError:
            print("Message NOT sent")
            
        time.sleep(0.1)