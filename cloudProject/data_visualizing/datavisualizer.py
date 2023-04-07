import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class datavisualizer:
    def shape(self, data):
        return data.shape
    def describe(self, data):
        return data.describe()
    def histogram(self, data):
        data.hist(figsize=(10,10))
        plt.savefig('graphs/histogram.PNG')
    def boxplot(self, data):
        lst = pd.Series.tolist(data.columns)
        plt.figure(figsize=(50, 50))
        data.boxplot(lst)
        plt.savefig('graphs/boxplot.PNG')