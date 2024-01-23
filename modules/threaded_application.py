import argparse
import threading
import queue
import time
from can_spi import MCP2515
import time
import torch
from pathlib import Path

import keras
from keras import layers

import numpy as np
import struct

parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
parser.add_argument('--trigger_buffer_size', type=int, default=70,
                    help='')
parser.add_argument('--past_timestep', type=int, default=30,
                    help='')
parser.add_argument('--data', type=str, default='battery',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='cmu.pkl',
                    help='filename of the dataset')

args = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
model = keras.models.load_model('autoencoder/weights/dl_autoencoder_tf.keras')
print("=> loaded checkpoint")

# Define a thread-safe queue for communication between threads
recv_buffer = queue.Queue()
send_buffer = queue.Queue()

# Initalize the CAN interface
can = MCP2515.MCP2515()
can.Init()

# Function to simulate receiving CAN signals and pushing into the buffer
def receive_can_signals():
    while True:
        # Simulate receiving CAN signals
        recv_buf = can.Receive()
        # Push the CAN signal into the buffer
        recv_buffer.put(recv_buf)


def decode_byte_to_signal(buffer):
    return struct.unpack("f", bytearray([int(i, 16) for i in buffer]))

# Function to perform AI processing on the CAN signals
def process_can_signals():
    while True:
        past_data = []
        if recv_buffer.qsize() > args.trigger_buffer_size:
            # Get the CAN signal from the buffer
            data = [decode_byte_to_signal(recv_buffer.get()) for i in range(0, min(recv_buffer.qsize(), args.trigger_buffer_size))]
            
            # Concat with data in the past for more relied predictions
            infering_data = past_data + data
            # past_data = data[-args.past_timestep:]
            
            # Perform AI processing on the CAN signal
            scores, predictions = perform_ai_processing(infering_data)
            
            # send_buffer.put(output_data)

# Function to perform AI processing on CAN signals (replace with your AI logic)
def perform_ai_processing(infering_data):
    X = np.array(infering_data)[np.newaxis, ...] #batch, seq_len, feature
    
    X_pred = model.predict(X)
    
    mae_loss = np.mean(np.abs(X - X_pred), axis=1).reshape((-1))
    return mae_loss, X_pred

def send_can_signals():
    pass

def main():
    # Create and start threads
    receive_thread = threading.Thread(target=receive_can_signals)
    process_thread = threading.Thread(target=process_can_signals)
    send_thread = threading.Thread(target=send_can_signals)

    receive_thread.start()
    process_thread.start()
    send_thread.start()

    # Wait for threads to finish (you can use asyncio.gather for asyncio)
    receive_thread.join()
    process_thread.join()
    send_thread.join()

if __name__ == "__main__":
    main()