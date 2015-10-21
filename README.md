# Kaggle's Flavours of Physics: application for the special "HEP meets ML" award

This solution scored 0.998150 on the [Private Leaderboard](https://www.kaggle.com/c/flavours-of-physics/leaderboard) of the Kaggle ["Flavours of Physics: Finding τ → μμμ"](https://www.kaggle.com/c/flavours-of-physics) competition. The model is based on ensemble of 20 feed-forward neural nets implemented with the help of the [Keras](https://github.com/fchollet/keras) library. In order to pass the [correlation test](https://www.kaggle.com/c/flavours-of-physics/details/correlation-test) and the [agreement test](https://www.kaggle.com/c/flavours-of-physics/details/agreement-test) the procedure of **Transfer learning** was implemented with help of additional "transductive" neural net (see [Application](docs/Application.pdf) for details).

## Dependencies

* The [Keras](https://github.com/fchollet/keras) and [Theano](http://deeplearning.net/software/theano/) libraries should be installed
* The standard Python packages **numpy**, **pandas**, **scipy**, **sklearn**, **h5py** and **cPickle** are required
* The training and test datasets (the files **training.csv** and **test.csv**), just like test files for control channel (the files **check_correlation.csv** and **check_agreement.csv**) can be downloaded from [here](https://www.kaggle.com/c/flavours-of-physics/data)
* Pretrained models and exemplary submissions can be found in **models** and **submission** directories

## How to generate the solution

 1. Set up your paths in **misc.py** and run **python misc.py** to create necessary directories on your disk
 2. Put the data files **training.csv**, **test.csv**, **check_correlation.csv** and **check_agreement.csv** in the **data** directory.
 3. To train the Keras classifier run **python model_keras.py**. The trained model will be saved in the **models/keras_model_20** directory, its predictions to **submission** and **transductor** directories.
 4. Run **python transductor_train.py** and let transductor to generate several models. This may take a while. See **transductor_train.py** for details.
 5. Run **python transductor_decorrelate_models.py** twice: first in 'dump' mode, second in 'decorrelate' mode. See **transductor_decorrelate_models.py** for details. The 'decorrelate' mode will generate submission, results will be written to **transductor_submission.csv**.
