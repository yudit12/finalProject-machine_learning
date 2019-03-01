from numpy import loadtxt
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 # load data
dataset = loadtxt('try.csv', delimiter=",")
