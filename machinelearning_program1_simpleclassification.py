from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import os  # Windows OS
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dataset = pd.read_csv("D:\Bits\Sem 1\ML\ML_data_sets\zoo.csv")
dataset.shape
print (dataset.head())
animal_names = dataset['animal_name'].tolist()
dataset=dataset.drop('animal_name',axis=1)
X = dataset.loc[:, dataset.columns != 'class_type']
Y = dataset['class_type']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, stratify=Y, test_size= 0.3)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
feature_names = list(dataset.columns.values)
feature_names = feature_names[:-1]
class_int = dataset['class_type'].unique().tolist()
class_names = ['Mammal', 'Fish', 'Bird', 'Invertebrate', 'Bug', 'Amphibian', 'Reptile']
dictionary = dict(zip(class_names, class_int))

dot_data = StringIO()

export_graphviz(model, out_file=dot_data,
                filled=True, rounded=True,
                feature_names = feature_names,
                class_names = class_names,
                proportion = False, precision = 2,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
