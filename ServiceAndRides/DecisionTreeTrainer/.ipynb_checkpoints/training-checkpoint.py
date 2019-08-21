import datetime
import os
import subprocess
import sys
from sklearn.externals import joblib

from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import pickle

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'nyc_servicerequest'

data_filename = 'x_all_train.csv'
target_filename = 'y_all_train.csv'
data_dir = 'gs://nyc_servicerequest/sklearnInput'

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir, data_filename), data_filename], stderr=sys.stdout)

subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir, target_filename), target_filename], stderr=sys.stdout)


"""
xs = pd.DataFrame()
ys = pd.DataFrame()

files = list(os.listdir('localsave2'))

for each in files:
    if('x_train' in each):
        a  = pd.read_csv(each, header=None).iloc[:, 1:]
        xs = pd.concat([xs, a])
    elif('y_train' in each):
        a = pd.read_csv(each, header=None)
        ys = pd.concat([ys, a])
    else:
        pass

"""
# Load data into pandas, then use `.values` to get NumPy arrays
data = pd.read_csv(data_filename).iloc[:, 1:].values
target = pd.read_csv(target_filename).iloc[:, 1:].values

# Convert one-column 2D array into 1D array for use with scikit-learn
target = target.reshape((target.size,))



#Train the model
dec = DecisionTreeRegressor()
dec.fit(data, target)

#Export the classifier to a file
model_filename = 'model.joblib'
joblib.dump(dec, model_filename)


with open('model.pkl', 'wb') as model_file:
    pickle.dump(dec, model_file)


# Upload the saved model file to Cloud Storage
gcs_model_path = os.path.join('gs://', BUCKET_NAME, 'models',
    datetime.datetime.now().strftime('decision_tree_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)