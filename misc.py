""" Set these paths and run this module to create them on disk
data_path
submission_path
temp_path
transductor_path
model_path
keras_model_path
"""

from os.path import join
from os.path import basename
import subprocess
import pandas as pd
import os


# Main paths
data_path = r'./data'
training_file = join(data_path, 'training.csv')
test_file = join(data_path, 'test.csv')
check_agreement_file = join(data_path, 'check_agreement.csv')
check_correlation_file = join(data_path, 'check_correlation.csv')


# Submission paths
submission_path = r'./submission'
xgboost_submission_file = join(submission_path, "xgboost_submission.csv")
keras_submission_file = join(submission_path, 'keras_submission.20.csv')
combined_submission_file = join(submission_path, 'combined_submission.0.3.20.csv')


# Temporary paths
temp_path = r'./temp'

# Model paths
model_path = r'./models/'
keras_model_path = join(model_path, r'./keras_model_20')


# transductor paths
transductor_path = r'./transductor'    
transductor_model_file = join(transductor_path, "transductor_model_{}.pkl")
transductor_scaler_file = join(transductor_path, "transductor_scaler.pkl")

# raw predictions
raw_submission_file = keras_submission_file
train_prediction_file = join(transductor_path, "train_prediction.csv")
check_agreement_prediction_file = join(transductor_path, "check_agreement_prediction.csv")
check_correlation_prediction_file = join(transductor_path, "check_correlation_prediction.csv")

transductor_submission_file = join(submission_path, "transductor_submission.csv")

# storage for transductor predictions
transductor_predictions_file = join(transductor_path, "transductor_predictions.hd5")
# preprocessed input
X_file = join(temp_path, "X_file.hd5")

transductor_pretrained_model_file = join(transductor_path, "transductor_pretrained_model.pkl")
transductor_log_file = join(transductor_path, "transductor.log")


def add_features(df):
    #significance of flight distance
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    df['NEW_IP_dira'] = df['IP']*df['dira']
    #Stepan Obraztsov's magic features
    df['NEW_FD_SUMP'] = df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt'] = df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    #some more magic features
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # "super" feature changing the result from 0.988641 to 0.991099
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df

    
def process_in_chunks(in_csv, fun, features, chunksize=100000):
    
    print("Processing {} in chunks".format(basename(in_csv)))

    # Get number of lines in the CSV file
    nlines = subprocess.check_output('wc -l %s' % in_csv, shell=True)
    nlines = int(nlines.split()[0])

    # Get header    
    header_row = pd.read_csv(in_csv, nrows=1).columns
    
    out_concat = None
    
    # Iteratively read CSV
    for k,i in enumerate(range(1, nlines, chunksize)):
        print("iteration {}, line {}".format(k, i))
    
        df = pd.read_csv(in_csv,  
                header=None,  # no header
                nrows=chunksize, # number of rows to read at each iteration
                names=header_row, # set header
                skiprows=i)   # skip rows that were already read
                
        df  = add_features(df)
        probs = fun(df[features])
        df_out = pd.DataFrame({"id": df["id"], "prediction": probs})
        out_concat = df_out if out_concat is None else pd.concat((out_concat, df_out))
    return out_concat


if __name__ == '__main__':
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)        
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)  
    if not os.path.exists(transductor_path):
        os.makedirs(transductor_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(keras_model_path):
        os.makedirs(keras_model_path)            