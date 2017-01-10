# dsl-char-cnn
Character-level CNN model for the [DSL 2016 shared task](http://ttg.uni-saarland.de/vardial2016/dsl2016.html).

## Requirements
* [Keras](https://keras.io)
* [TensorFlow](https://www.tensorflow.org). Not tested with Theano, but should work fine except for some imports in the beginning of the code. 
* Cuda GPU card will make the code much faster, especially with cudnn.

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
This will create a file with predictions under `data`. It will also create a file with the posterior probabilitites. See example files under `data`. 

Note: file names are currently hard-coded in several places (e.g. model files in the train and test scripts, and data files in `data.py`.

## Citating
If you use this code in your work, please consider citing our [paper](http://aclweb.org/anthology/W/W16/W16-4819.pdf): "A Character-level Convolutional Neural Network for Distinguishing Similar Languages and Dialects", Yonatan Belinkov and James Glass, VarDial 2016.

```bib
@InProceedings{belinkov-glass:2016:VarDial,
  author    = {Belinkov, Yonatan  and  Glass, James},
  title     = {A Character-level Convolutional Neural Network for Distinguishing Similar Languages and Dialects},
  booktitle = {Proceedings of the Third Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial)},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan}
}
```

### TODO
* Clean code

