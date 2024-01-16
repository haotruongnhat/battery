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
parser = argparse.ArgumentParser(description='PyTorch RNN Anomaly Detection Model')
parser.add_argument('--prediction_window_size', type=int, default=10,
                    help='prediction_window_size')
parser.add_argument('--data', type=str, default='ecg',
                    help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl',
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
print("=> loaded checkpoint")


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data,filename=args.filename, augment_test_data=False)
train_dataset = TimeseriesData.batchify(args,TimeseriesData.trainData[:TimeseriesData.length], bsz=args.batch_size)
test_dataset = TimeseriesData.batchify(args,TimeseriesData.testData, bsz=args.batch_size)

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
try:
    # For each channel in the dataset
    for channel_idx in range(nfeatures):
        ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
        # Mean and covariance are calculated on train dataset.
        if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
            print('=> loading pre-calculated mean and covariance')
            mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
        else:
            print('=> calculating mean and covariance')
            mean, cov = fit_norm_distribution_param(args, model, train_dataset, channel_idx=channel_idx)

        ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
        # An anomaly score predictor is trained
        # given hidden layer output and the corresponding anomaly score on train dataset.
        # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
        if args.compensate:
            print('=> training an SVR as anomaly score predictor')
            train_score, _, _, hiddens, _ = anomalyScore(args, model, train_dataset, mean, cov, channel_idx=channel_idx, batch_size=args.batch_size)
            score_predictor = GridSearchCV(SVR(), cv=5,param_grid={"C": [1e0, 1e1, 1e2],"gamma": np.logspace(-1, 1, 3)})
            score_predictor.fit(torch.cat(hiddens,dim=0).numpy(), train_score.cpu().numpy())
        else:
            score_predictor=None

        ''' 3. Calculate anomaly scores'''
        # Anomaly scores are calculated on the test dataset
        # given the mean and the covariance calculated on the train dataset
        print('=> calculating anomaly scores')
        import time; t=time.time()
        score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(args, model, test_dataset, mean, cov,
                                                                                  score_predictor=score_predictor,
                                                                                  channel_idx=channel_idx,
                                                                                  batch_size=args.batch_size)
        print(time.time() - t)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
