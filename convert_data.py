import pandas as pd
import pickle
import numpy as np

data1 = pd.read_csv("VAH01.csv", usecols=['Ecell_V'])[:110000]
data1_np = data1.to_numpy()
zero = np.zeros((data1_np.shape[0],1))
labelled_data1 = np.hstack((data1_np, zero))

with open(r'modules\anomaly\dataset\battery\labeled\train\cmu.pkl', 'wb') as b:
    pickle.dump(labelled_data1.tolist(), b)

data2 = pd.read_csv("VAH05.csv", usecols=['Ecell_V'])[:50000]
data2_np = data2.to_numpy()
zero = np.zeros((data2_np.shape[0],1))
labelled_data2 = np.hstack((data2_np, zero))

with open(r'modules\anomaly\dataset\battery\labeled\test\cmu.pkl', 'wb') as b:
    pickle.dump(labelled_data2.tolist(), b)