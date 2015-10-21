'''
run this script until enough transductor models get dumped.
'enough' is any number reasonable for you.
quality increases with quantity, I dumped 300, but 1-2 is okay too
'''

import numpy as np
import pandas as pd
import misc as pt
from misc import add_features
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from evaluation import roc_auc_truncated, compute_ks, compute_cvm
import cPickle
from scipy.optimize import minimize
import logging
            
np.random.seed(1337) # for reproducibility

logger = logging.getLogger()
hdlr = logging.FileHandler(pt.transductor_log_file)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

    
def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X[:,:-1])    # don't scale last col - prediction
    X[:,:-1] = scaler.transform(X[:,:-1])
    return X, scaler

    
def load(data_file, prediction_file, tail=None, weight=False, mass=False):
    data = pd.read_csv(data_file)
    data = add_features(data)
    prediction = pd.read_csv(prediction_file)
    data['prediction'] = prediction["prediction"]
    if tail is not None:
        data = data[-tail:]

    # shuffle
    data = data.iloc[np.random.permutation(len(data))].reset_index(drop=True)

    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'weight', 'signal']
    features = list(f for f in data.columns if f not in filter_out)
    X = data[features].values
    y = data['signal'].values if not mass else None
    w = data['weight'].values if weight else None
    m = data['mass'].values if mass else None
    return X, y, w, m


def create_model(input_shape):
    np.random.seed(11) # for reproducibility
    
    model = Sequential()
    model.add(Dense(input_shape, 50))
    model.add(Activation('tanh'))
    
    model.add(Dense(50, 50))
    model.add(Activation('tanh'))
    
    model.add(Dense(50, 30))
    model.add(Activation('tanh'))    
    
    model.add(Dense(30, 25))
    model.add(Activation('tanh'))
    
    model.add(Dense(25, 2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def get_weights(model):
    weights = model.get_weights()
    return np.concatenate([x.ravel() for x in weights])


def set_weights(model, parameters):
    weights = model.get_weights()
    start = 0
    for i,w in enumerate(weights):
        size = w.size
        weights[i] = parameters[start:start+size].reshape(w.shape)
        start += size
    model.set_weights(weights)
    return model


def dump_transductor_model(model, transductor_model_file):
    with open(transductor_model_file, 'wb') as fid:
        cPickle.dump(model, fid)
    
    
def create_objective(model, transductor_model_file, X, y, Xa, ya, wa, Xc, mc, 
                     ks_threshold=0.09, cvm_threshold=0.002, verbose=True):
    i = []
    d = []
    auc_log = [0]

    def objective(parameters):
        i.append(0)
        set_weights(model, parameters)
        p = model.predict(X, batch_size=256, verbose=0)[:, 1]
        auc = roc_auc_truncated(y, p)

        pa = model.predict(Xa, batch_size=256, verbose=0)[:, 1]
        ks = compute_ks(pa[ya == 0], pa[ya == 1], wa[ya == 0], wa[ya == 1])
        
        pc = model.predict(Xc, batch_size=256, verbose=0)[:, 1]        
        cvm = compute_cvm(pc, mc)

        ks_importance = 1  # relative KS importance
        ks_target = ks_threshold
        cvm_importance = 1  # relative CVM importance
        cvm_target = cvm_threshold
        
        alpha = 0.001        # LeakyReLU
        ks_loss = (1 if ks > ks_target else alpha) * (ks - ks_target)
        cvm_loss = (1 if cvm > cvm_target else alpha) * (cvm - cvm_target)
        loss = -auc + ks_importance*ks_loss + cvm_importance*cvm_loss        

        if ks < ks_threshold and cvm < cvm_threshold and auc > auc_log[0]:
            d.append(0)
            dump_transductor_model(model, transductor_model_file.format(len(d)))
            auc_log.pop()
            auc_log.append(auc)
            message = "iteration {:7}: Best AUC={:7.5f} achieved, KS={:7.5f}, CVM={:7.5f}".format(len(i), auc, ks, cvm)
            logger.info(message)

        if verbose:
            print("iteration {:7}: AUC: {:7.5f}, KS: {:7.5f}, CVM: {:7.5f}, loss: {:8.5f}".format(len(i), 
                  auc, ks, cvm, loss))
        return loss
    return objective
    
    
Xt, yt, _, _ = load(pt.training_file, pt.train_prediction_file)    # shuffled
Xa, ya, wa, _ = load(pt.check_agreement_file, pt.check_agreement_prediction_file,
                     tail=len(yt), weight=True)
Xc, yc, _, mc = load(pt.check_correlation_file, pt.check_correlation_prediction_file,
                     mass=True)
Xt, scaler = preprocess_data(Xt)
Xa = preprocess_data(Xa, scaler)[0]
Xc = preprocess_data(Xc, scaler)[0]
with open(pt.transductor_scaler_file, 'wb') as fid:
    cPickle.dump(scaler, fid)   

AUC = roc_auc_truncated(yt, Xt[:,-1])
print ('AUC before transductor', AUC)  

model = create_model(Xt.shape[1])

pretrain = True
if pretrain:
    # pretrain model
    print("Pretrain model")
    yt_categorical = np_utils.to_categorical(yt, nb_classes=2)
    model.fit(Xt, yt_categorical, batch_size=64, nb_epoch=1,
              validation_data=None, verbose=2, show_accuracy=True)
    print("Save pretrained model")
    with open(pt.transductor_pretrained_model_file, 'wb') as fid:
        cPickle.dump(model, fid)
else:
    print("Load pretrained model")
    with open(pt.transductor_pretrained_model_file, 'rb') as fid:
        model = cPickle.load(fid)

x0 = get_weights(model)
print("Optimize %d weights" % len(x0))
objective = create_objective(model, pt.transductor_model_file,
                             Xt, yt, Xa, ya, wa, Xc, mc, verbose=True)
minimize(objective, x0, args=(), method='Powell')
