---
title: snippets
date: 2022-05-04 22:45:18
tags: [Snippet,Python]
categories: Snippet
---

Sklearn的dataset转为dataframe
``` python
import pandas as pd
from sklearn import datasets

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['TARGET'] = pd.Series(sklearn_dataset.target)
    return df

df_boston = sklearn_to_df(datasets.load_boston())

print(df_boston.head())
```