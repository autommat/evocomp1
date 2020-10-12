import pandas as pd
from sklearn import preprocessing

df=pd.read_csv("res/iris.data", header=None)
df_attributes = df.iloc[:, :-1]
x = df_attributes.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_attributes_normalized = pd.DataFrame(x_scaled)
df_attributes_normalized.to_csv('normalizedIris.csv', header=None)