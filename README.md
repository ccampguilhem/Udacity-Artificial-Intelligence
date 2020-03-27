# Udacity - Artificial Intelligence Nanodegree

Cédric Campguilhem, 2020

## Projects

Project | Description                     | Link
--------|---------------------------------|------- 
01      | Deep learning: image classifier | ./02_Deep_Neural_Networks/P01_Image_Classifier/Image_Classifier_Project.ipynb

## Data Engineering

### ETL pipeline

#### pandas

You can access specific method of datetime objects in pandas dataframe by 
using `df[column].dt.???`

The following code collects all the encodings available on the system:

```python
from encodings.aliases import aliases
alias_values = set(aliases.values())
```

Then, you can use:

```python
import pandas as pd
pd.read_csv(filename, encoding=alias)
```

Or, you can use [chardet](https://pypi.org/project/chardet/) library:

```python
# import the chardet library
import chardet 

# use the detect method to find the encoding
# 'rb' means read in the file as binary
with open("mystery.csv", 'rb') as file:
    print(chardet.detect(file.read()))
```

Pandas provide a method [get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) to do one-hot encoding:

```
>>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
   b  c
0  0  0
1  1  0
2  0  1
3  0  0
4  0  0
```

#### scikit-learn

Here is documentation on outlier processing with [sklearn](https://scikit-learn.org/stable/modules/outlier_detection.html).
The following explanation [here](https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561) is also very thorough.
Also look [here](https://en.wikipedia.org/wiki/Outlier#Tukey's_fences) for Tukey's fences method.

#### Other libraries

Library [pycountry](https://pypi.org/project/pycountry/) provides standards 
for countries.

A tutorial for the use of [Regular Expressions](https://medium.com/factory-mind/regex-tutorial-a-simple-cheatsheet-by-examples-649dc1c3f285).

### NLP Pipelines

#### Word embedding techniques

[Word2Vec](https://www.youtube.com/watch?time_continue=9&v=7jjappzGRe0&feature=emb_logo)

[GloVe](https://www.youtube.com/watch?v=KK3PMIiIn8o&feature=emb_logo)

[Uses](https://www.youtube.com/watch?time_continue=2&v=gj8u1KG0H2w&feature=emb_logo) in Deep Learning.

Visualization of word embedding with [t-SNE](https://www.youtube.com/watch?v=xxcK8oZ6_WE&feature=emb_logo)

### Machine Learning Pipelines

You can implement your own Estimator, Classifier or Transformer with 
[Scikit-Learn](https://scikit-learn.org/stable/developers/develop.html?highlight=custom%20transformer)

There is also a [class](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer) to create a transformer from an existing function.

## Deep Learning: Neural Networks

### Udacity GitHub repos

https://github.com/udacity/DSND_Term1


### Gradient Descent

Using [momentum](https://distill.pub/2017/momentum/) with gradient descent to avoid falling into local minimum.

[Optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#rmsprop)

An [overview](https://ruder.io/optimizing-gradient-descent/index.html) of gradient descent algorithms.

### Backpropagation algorithm

Backpropagation algorithm explained [here](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9) and [there](https://www.youtube.com/watch?v=59Hbtz7XgjM) by Andrej Karpathy.

### PyTorch

PyTorch [installation](https://pytorch.org/get-started/locally/).

PyTorch API [documentation](https://pytorch.org/docs/stable/torch.html).

### Keras

[Optimizers](https://keras.io/optimizers/)

Training [progress](https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training-process-in-keras/) with Keras.

### TensorFlow

How to install with [conda](https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc).

[Migrating](https://www.tensorflow.org/guide/migrate) TensorFlow 1.x to 2.x. Which is a pain in the neck, the better option is to start working with TensorFlow 2.x using tf.keras instead of Keras alone. Check [here](https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/) for more information.

Official [tutorials](https://www.tensorflow.org/tutorials).

[Custom](https://towardsdatascience.com/custom-loss-function-in-tensorflow-2-0-d8fa35405e4e) loss function.

YouTube tutorial by [freeCodeCamp.org](https://www.youtube.com/watch?v=tPYj3fFJGjk).

### CUDA

We can use the `nvidia-smi` command to monitor the GPU usage.

To [de-activate](https://datascience.stackexchange.com/questions/58845/how-to-disable-gpu-with-tensorflow) CUDA to avoid GPU processing.

### Loss functions

When to use [cross-entropy](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy).

[Cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) function.


