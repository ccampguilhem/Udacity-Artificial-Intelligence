# Udacity - Artificial Intelligence Nanodegree

Cédric Campguilhem, 2020

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


