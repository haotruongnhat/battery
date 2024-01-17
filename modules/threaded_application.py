import argparse
import threading
import queue
import time
from can_spi import MCP2515
import time
import torch
from anomaly.anomalyDetector import anomalyScore
from anomaly.model import model
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
parser.add_argument('--trigger_buffer_size', type=int, default=4,
                    help='')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--queue_max_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
                    help='filename of the dataset')

args_ = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
checkpoint = torch.load(str(Path('anomaly','save',args_.data,'checkpoint',args_.filename).with_suffix('.pth')))
args = checkpoint['args']
args.prediction_window_size= args_.prediction_window_size
args.trigger_buffer_size= args_.trigger_buffer_size
args.batch_size= args_.batch_size

print("=> loaded checkpoint")

nfeatures = 8
model = model.RNNPredictor(rnn_type = args.model,
                           enc_inp_size=nfeatures,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=nfeatures,
                           nlayers = args.nlayers,
                           res_connection=args.res_connection).to(args.device)
model.load_state_dict(checkpoint['state_dict'])

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
        print("Message received")
        # Push the CAN signal into the buffer
        recv_buffer.put(recv_buf)

# Function to perform AI processing on the CAN signals
def process_can_signals():
    while True:
        if recv_buffer.qsize() > args.trigger_buffer_size:
            # Get the CAN signal from the buffer
            can_signal = recv_buffer.get()
            data = [recv_buffer.get() for i in range(0, min(recv_buffer.qsize(), args.trigger_buffer_size))]
            # Perform AI processing on the CAN signal
            # output_data = perform_ai_processing(can_signal)
            print(data)
            # send_buffer.put(output_data)

# Function to perform AI processing on CAN signals (replace with your AI logic)
def perform_ai_processing(can_signal):
    # Simulate AI processing logic
    processed_data = can_signal["data"].upper()
    return processed_data

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