# Tensorflow talk example

A copy of the [text classification example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/skflow/text_classification_cnn.py)
provided by the tensorflow distribution.  This is a straightforward extension to include some non standard locations/files and modification of
hyperparameters.  A driver program is included to work around some bugs in the tensorflow implementation.

Training is invoked by

    ./driver.py jobX1_classification.py 

The file `embeddings_ops.py` is lifted out of **skflow** to show the implementation of **categorical_variable**.
