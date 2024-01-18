import sklearn
import argparse
import torch
import pickle
import preprocess_data
from model import model
from torch import optim
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from anomalyDetector import fit_norm_distribution_param
from anomalyDetector import anomalyScore
from anomalyDetector import get_precision_recall
import time

parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
parser.add_argument('--data', type=str, default='battery',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='cmu.pkl',
                    help='filename of the dataset')
parser.add_argument('--save_fig', action='store_true',
                    help='save results as figures')
parser.add_argument('--compensate', action='store_true',
                    help='compensate anomaly score using anomaly score esimation')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta value for f-beta score')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')

args_ = parser.parse_args()
print('-' * 89)
print("=> loading checkpoint ")
checkpoint = torch.load(str(Path('save',args_.data,'checkpoint',args_.filename).with_suffix('.pth')))
args = checkpoint['args']
args.prediction_window_size= args_.prediction_window_size
args.beta = args_.beta
args.save_fig = args_.save_fig
args.compensate = args_.compensate
args.batch_size = args_.batch_size
print("=> loaded checkpoint")


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data,filename=args.filename, augment_test_data=False)
train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData[:TimeseriesData.length], bsz=args.batch_size)
test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData[:1000], bsz=args.batch_size)

###############################################################################
# Build the model
###############################################################################
nfeatures = TimeseriesData.trainData.size(-1)
model = model.RNNPredictor(rnn_type = args.model,
                           enc_inp_size=nfeatures,
                           rnn_inp_size = args.emsize,
                           rnn_hid_size = args.nhid,
                           dec_out_size=nfeatures,
                           nlayers = args.nlayers,
                           res_connection=args.res_connection).to(args.device)
model.load_state_dict(checkpoint['state_dict'])
#del checkpoint

scores, predicted_scores, precisions, recalls, f_betas = list(), list(), list(), list(), list()
targets, mean_predictions, oneStep_predictions, Nstep_predictions = list(), list(), list(), list()
# For each channel in the dataset
for channel_idx in range(nfeatures):
    ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
    # Mean and covariance are calculated on train dataset.
    mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
    score_predictor=None
    
    t = time.time()
    score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(args, model, test_dataset, mean, cov,
                                                                                score_predictor=score_predictor,
                                                                                channel_idx=channel_idx,
                                                                                batch_size=args.batch_size)
    print(time.time() - t, "s")
    target = preprocess_data.reconstruct(test_dataset.cpu()[:, 0, channel_idx],
                                             TimeseriesData.mean[channel_idx],
                                             TimeseriesData.std[channel_idx]).numpy()
    Nstep_prediction = preprocess_data.reconstruct(sorted_prediction[:, 0].cpu(),
                                                    TimeseriesData.mean[channel_idx],
                                                    TimeseriesData.std[channel_idx]).numpy()
    print(target.shape)
    score = score.cpu()
    save_dir = Path('result',args.data,args.filename).with_suffix('').joinpath('fig_detection')
    save_dir.mkdir(parents=True,exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(15,5))
    ax1.plot(target,label='Target',
                color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
    ax1.plot(Nstep_prediction, label=str(args.prediction_window_size) + '-step predictions',
                color='blue', marker='.', linestyle='--', markersize=1, linewidth=0.5)
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Value',fontsize=15)
    ax1.set_xlabel('Index',fontsize=15)
    ax2 = ax1.twinx()
    ax2.plot(score.numpy().reshape(-1, 1), label='Anomaly scores from \nmultivariate normal distribution',
                color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('anomaly score',fontsize=15)
    #plt.axvspan(2830,2900 , color='yellow', alpha=0.3)
    plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.xlim([0,len(test_dataset)])
    plt.savefig(str(save_dir.joinpath('fig_scores_channel'+str(channel_idx)).with_suffix('.png')))
    #plt.show()
    plt.close()
