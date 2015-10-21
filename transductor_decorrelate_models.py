'''
1. first, run this script in mode = 'dump' to dump predictions
from transductor models

set model range to include available transductor models, e.g. for
transductor_model_281.pkl through transductor_model_300.pkl range(281,301)

2. second, run this script in mode = 'decorrelate' to create combined 
submission

n_models_to_combine are the first least correlated models to combine.
Makes little effect in practice, 20 is better, but 1 is enough.
'''

import numpy as np
import pandas as pd
from os.path import basename
import subprocess
import cPickle
import h5py
import misc as pt
from misc import add_features

np.random.seed(1337) # for reproducibility


def preprocess_and_dump_hd5(in_csv, pf_csv, X_file, scaler, chunksize=100000):
    # Get number of lines in the CSV file
    nlines = subprocess.check_output('wc -l %s' % in_csv, shell=True)
    nlines = int(nlines.split()[0])

    # Get header
    df = pd.read_csv(in_csv, nrows=1) 
    header_row = df.columns
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    df  = add_features(df)
    features = list(f for f in df.columns if f not in filter_out)
    
    with h5py.File(X_file, "w") as f:
        X = f.create_dataset("X", (nlines-1,len(features)+1), dtype='float64')
        # Iteratively read CSV
        for k,i in enumerate(range(1, nlines, chunksize)):
            print("iteration {}, line {}".format(k, i))
        
            df = pd.read_csv(in_csv,  
                    header=None,  # no header
                    nrows=chunksize, # number of rows to read at each iteration
                    names=header_row, # set header
                    skiprows=i)   # skip rows that were already read
            pf = pd.read_csv(pf_csv,  
                    header=None,  # no header
                    nrows=chunksize, # number of rows to read at each iteration
                    names=["id", "prediction"], # set header
                    skiprows=i)   # skip rows that were already read
            df  = add_features(df)
            data = scaler.transform(df[features])
            data = np.hstack((data, pf['prediction'].values.reshape(-1,1)))
            X[i-1:i-1+chunksize,:] = data


model_range = range(281,301)
n_models_to_combine = 10

mode = 'decorrelate'   # dump (predictions) | decorrelate (and generate submission)
if mode == 'dump':
    print("Load transductor scaler")
    with open(pt.transductor_scaler_file, 'rb') as fid:
        scaler = cPickle.load(fid)

    print("Scale and dump test data to {}".format(basename(pt.X_file)))
    preprocess_and_dump_hd5(pt.test_file, pt.raw_submission_file, pt.X_file, scaler)
    
    with h5py.File(pt.X_file, "r") as f:
        X = f.get("X")
        with h5py.File(pt.transductor_predictions_file, "w") as fpreds:
            for k,i in enumerate(model_range):
                m = pt.transductor_model_file.format(i)
                print("Predict model {}/{} from {}".format(k+1,
                      len(model_range), basename(m)))
                with open(m, 'rb') as fid:
                    model = cPickle.load(fid)
                p = model.predict(X, batch_size=256, verbose=0)[:, 1]
                if k==0:    # 1st iteration
                    probs = fpreds.create_dataset("probs", (len(model_range),len(p)))
                probs[k,:] = np.array(p)
elif mode == 'decorrelate':
    print("Generate transductor submission")
    with h5py.File(pt.transductor_predictions_file, "r") as fpreds:
        probs = fpreds.get("probs")
        corr = np.corrcoef(probs)
        cr=corr.sum(axis=0)
        idx = np.argsort(cr)   # sorted correlations
        ids = idx[:n_models_to_combine]  # take first least corrlated
        ids.sort()             # h5py requires increasing index
        print("Models to combine: %s" % 
        ", ".join(map(lambda x: basename(pt.transductor_model_file.format(x)).split(".")[0], np.array(model_range)[ids])))
        p = probs[list(ids)].mean(axis=0)
    df = pd.read_csv(pt.test_file, usecols=["id"])
    transductor_submission = pd.DataFrame({"id": df["id"], "prediction": p})    
    transductor_submission.to_csv(pt.transductor_submission_file, index=False)             
else:
    print("Unknown mode")             
