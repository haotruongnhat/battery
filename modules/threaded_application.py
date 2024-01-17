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

import anomaly.preprocess_data as preprocess_data
import struct

class PickleLoad(preprocess_data.PickleDataLoad):
    def __init__(self, data_type, filename, augment_test_data=True):
        self.augment_test_data=augment_test_data
        self.trainData, self.trainLabel = self.preprocessing(Path('anomaly', 'dataset',data_type,'labeled','train',filename),train=True)
        self.testData, self.testLabel = self.preprocessing(Path('anomaly', 'dataset',data_type,'labeled','test',filename),train=False)

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

args_ = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
checkpoint = torch.load(str(Path('anomaly','save',args_.data,'checkpoint',args_.filename).with_suffix('.pth')))
args = checkpoint['args']
args.prediction_window_size= args_.prediction_window_size
args.trigger_buffer_size= args_.trigger_buffer_size

print("=> loaded checkpoint")

TimeseriesData = PickleLoad(data_type=args.data,filename=args.filename, augment_test_data=False)

nfeatures = 1
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
            past_data = data[-args.past_timestep:]
            
            # Perform AI processing on the CAN signal
            scores, predictions = perform_ai_processing(infering_data)
            
            # send_buffer.put(output_data)

# Function to perform AI processing on CAN signals (replace with your AI logic)
def perform_ai_processing(infering_data):
    data = TimeseriesData.batchify(args, infering_data, bsz=1)
    score_predictor=None

    scores = []
    predictions = []
    for channel_idx in range(nfeatures):
        mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
        score, sorted_prediction, _, _, _ = anomalyScore(args, model, data, mean, cov,
                                                        score_predictor=score_predictor,
                                                        channel_idx=channel_idx,
                                                        batch_size=1)
        target = preprocess_data.reconstruct(data.cpu()[:, 0, channel_idx],
                                             TimeseriesData.mean[channel_idx],
                                             TimeseriesData.std[channel_idx]).numpy()
        Nstep_prediction = preprocess_data.reconstruct(sorted_prediction[:, 0].cpu(),
                                                        TimeseriesData.mean[channel_idx],
                                                        TimeseriesData.std[channel_idx]).numpy()
        score = score.cpu()
        scores.append(score)
        predictions.append(sorted_prediction)

    return scores, predictions

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