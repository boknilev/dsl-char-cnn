# dsl-char-cnn
Character-level CNN model for the [DSL 2016 shared task](http://ttg.uni-saarland.de/vardial2016/dsl2016.html).

## Requirements
* [Keras](https://keras.io)
* [TensorFlow](https://www.tensorflow.org). Not tested with Theano, but should work fine except for some imports in the beginning of the code. 
* Cuda GPU card will make the code much faster, especially with cudnn

## Usage
To train a model, go to the `src` dir and run:
```
python cnn_multifilter.py
```
This will train a model, save it to disk, and report some scores.

to test a model on raw texts, go to the `src` dir and run:
```
python predict_test_data_with_trained_model.py
```
this will create a file with predictions under `data`. It will also create a file with the posterior probabilitites. See example files under `data`. 

Note: file names are currently hard-coded in several places (e.g. model files in the train and test scripts, and data files in `data.py`.

### TODO
* Clean code

