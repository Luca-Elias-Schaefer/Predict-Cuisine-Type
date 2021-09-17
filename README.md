# Predict-Cuisine-Type

In this project, we analyze different cuisines and recipes based on their ingredients. 
After an exploratory data analysis, an analysis of the similarities of different cuisines and a market basket analysis, we create a model to predict cuisines based on ingredients.

# Libraries
- import pandas as pd
- import numpy as np
- import seaborn as sns
- import matplotlib.pyplot as plt
- from IPython.core.display import HTML
- import nltk
- from nltk.corpus import stopwords
- import re
- from sklearn.feature_extraction.text import TfidfVectorizer
- from sklearn.metrics.pairwise import cosine_similarity
- from mlxtend.frequent_patterns import apriori
- from mlxtend.preprocessing import TransactionEncoder
- from sklearn.preprocessing import LabelEncoder
- from sklearn.model_selection import train_test_split
- from sklearn import metrics
- from sklearn.ensemble import AdaBoostClassifier
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.neighbors import KNeighborsClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.linear_model import LogisticRegression

# Datasets
- ingredients.csv
- recipes.csv
