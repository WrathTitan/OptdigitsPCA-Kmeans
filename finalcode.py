#Applying PCA to Optdigits dataset and selecting features that explain 90% of the variance and Clustering with Kmeans and reporting accuracy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random as rd

optdigits_df=pd.read_csv('optdigits_csv.csv') #Loading the dataset from the csv file

data_features=optdigits_df.iloc[:,:64] #Here we are selecing the first 64 columns which contains our data

data_labels=optdigits_df.iloc[:,64:] #Here we are selecing the last column which contains the labels associated with our data

