import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from os.path import join, basename
import cPickle
import misc as pt
from misc import add_features
import h5py
import subprocess

np.random.seed(1337) # for reproducibility


def load_data(train_file, shuffle=False, seed=None):
    
    print("Load the training data")
    df = pd.read_csv(train_file)
    df = add_features(df)
    if shuffle:
        if seed is not None:
            np.random.seed(seed) # seed to shuffle the train set
        df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal']
    features = list(f for f in df.columns if f not in filter_out)
    return df[features].values, df['signal'].values, features


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_and_dump_hd5(in_csv, X_file, scaler, chunksize=100000):
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
        X = f.create_dataset("X", (nlines-1,len(features)), dtype='float64')
        # Iteratively read CSV
        for k,i in enumerate(range(1, nlines, chunksize)):
            print("iteration {}, line {}".format(k, i))
            df = pd.read_csv(in_csv,  
                    header=None,  # no header
                    nrows=chunksize, # number of rows to read at each iteration
                    names=header_row, # set header
                    skiprows=i)   # skip rows that were already read
            df  = add_features(df)
            data = scaler.transform(df[features])
            X[i-1:i-1+chunksize,:] = data


def save_prediction(classifier, in_file, out_file):
        probs = classifier.predict_proba(in_file)[:,1]
        df = pd.read_csv(in_file, usecols=["id"])
        submission = pd.DataFrame({"id": df["id"], "prediction": probs})    
        submission.to_csv(out_file, index=False)

    
def model_factory(n_inputs):
    model = Sequential()
    model.add(Dense(n_inputs, 75))
    model.add(PReLU((75,)))
    
    model.add(Dropout(0.11))
    model.add(Dense(75, 50))
    model.add(PReLU((50,)))
    
    model.add(Dropout(0.09))
    model.add(Dense(50, 30))
    model.add(PReLU((30,)))
    
    model.add(Dropout(0.07))
    model.add(Dense(30, 25))
    model.add(PReLU((25,)))
    
    model.add(Dense(25, 2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


class KerasClassifier(object):
    
    def __init__(self, split=None, n_models=10, n_epoch=85, features=None,
                 model_factory=model_factory):
        self.split = split
        self.n_models = n_models
        self.n_epoch = n_epoch
        self.features = features
        self.scaler = None
        self.models = []
        self.model_factory = model_factory
        
    def __str__(self):
        return "KerasClassifier(split={}, n_models={}, n_epoch={})".\
        format(self.split, self.n_models, self.n_epoch)

    def __repr__(self):
        return "KerasClassifier(split={}, n_models={}, n_epoch={})".\
        format(self.split, self.n_models, self.n_epoch)
        
    def fit(self, X, y):
        self.models = []
        # preprocess the data
        X, self.scaler = preprocess_data(X)
        y = np_utils.to_categorical(y)

        # split into training / evaluation data
        if self.split is not None:
            nb_train_sample = int(len(y) * self.split)
            X_train = X[:nb_train_sample]
            X_eval = X[nb_train_sample:]
            y_train = y[:nb_train_sample]
            y_eval = y[nb_train_sample:]
            validation_data=(X_eval, y_eval)
            print('Train on:', X_train.shape[0])
            print('Eval on:', X_eval.shape[0])
        else:
            X_train = X
            y_train = y
            validation_data = None
            print('No evaluation')
        
        n_inputs = X.shape[1]
        for i in range(self.n_models):
            print("\n----------- Keras: train Model %d/%d ----------\n" % (i+1,self.n_models))
            model = self.model_factory(n_inputs)
            model.fit(X_train, y_train, batch_size=64, nb_epoch=self.n_epoch,
                      validation_data=validation_data, verbose=2,
                      show_accuracy=True)
            self.models.append(model)
        return self
    
    def predict_proba(self, X):

        def models_loop(X):
            probs = None
            for i,model in enumerate(self.models):
                print("----------- Keras: predict Model %d/%d ----------" % (i+1,len(self.models)))
                p = model.predict(X, batch_size=256, verbose=0)[:, 1]
                probs = p if probs is None else probs + p
            return probs
        
        if isinstance(X, str):
            print("\nScale and dump {} to {}".format(basename(X), basename(pt.X_file)))
            preprocess_and_dump_hd5(X, pt.X_file, self.scaler)
            with h5py.File(pt.X_file, "r") as f:
                X = f.get("X")        
                probs = models_loop(X)
        else:
            X = preprocess_data(X, scaler=self.scaler)[0]
            probs = models_loop(X)
        r = np.zeros((len(probs),2))
        r[:,1] = probs / len(self.models)
        return r

    def save_model(self, model_path, model_prefix="model{}.pkl"):
        print("Saving model to %s" % model_path)
        for i,model in enumerate(self.models):
            model_file = join(model_path, model_prefix.format(i))
            with open(model_file, 'wb') as fid:
                cPickle.dump(model, fid)


if __name__ == '__main__':

    # load data and train model    
    X, y, features = load_data(pt.training_file, shuffle=True, seed=1337)
    cls = KerasClassifier(split=None, n_models=20, n_epoch=100, features=features)
    cls.fit(X, y)

    # save model
    cls.save_model(pt.keras_model_path) 

    # make prediction on test, train, agreement and correlation files
    save_prediction(cls, pt.test_file, pt.keras_submission_file)
    save_prediction(cls, pt.training_file, pt.train_prediction_file)
    save_prediction(cls, pt.check_agreement_file, pt.check_agreement_prediction_file)
    save_prediction(cls, pt.check_correlation_file, pt.check_correlation_prediction_file)
