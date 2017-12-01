import pandas
import graphviz
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

names1 = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
class1 = ['setosa', 'versicolor', 'virginica']
dataset = pandas.read_csv('iris.data', names=names1)

print("Tamanho da base e quantidade de atributos: ")
print("==========================================")
print(dataset.shape)

print("\nAtributos com as 20 primeiras linhas da base: ")
print("=============================================")
print(dataset.head(20))

print("\nDescricao estatística da base: ")
print("==============================")
print(dataset.describe())

print("\nClasse e distribuicao: ")
print("======================")
print(dataset.groupby('class').size())

# 80% do dataset para treinamento e 20% para teste
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7
scoring = 'accuracy'

models = []
models.append(('Arvore de Decisao', DecisionTreeClassifier()))

results = []
names = []
for name, model in models:
    # Utilizacao do 10-fold cross validation
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "\n%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

dectree = DecisionTreeClassifier()
dectree.fit(X_train, Y_train)
predictions = dectree.predict(X_validation)
print("\nAcuracia: %f" % accuracy_score(Y_validation, predictions))
print("\nMatriz de Confusao: ")
print(confusion_matrix(Y_validation, predictions))
print("\nMetricas da classificacao: ")
print("==========================")
print(classification_report(Y_validation, predictions))

dot_data = tree.export_graphviz(dectree, out_file=None, feature_names=names1, class_names=class1, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")
from subprocess import check_call
check_call(['dot','-Tpng','iris','-o','iris.png'])
from PIL import Image
image = Image.open("iris.png")
image.show()
