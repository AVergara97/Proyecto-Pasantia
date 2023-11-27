import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder,StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.cluster import KMeans
from sklearn.tree import export_text
#import seaborn as sns
import re
import joblib