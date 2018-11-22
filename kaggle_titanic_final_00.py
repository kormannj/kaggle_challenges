import pandas as pd

#Importar os dados de treino e de teste
base_treinamento = pd.read_csv('train.csv')
base_teste = pd.read_csv('test.csv')

#Excluir colunas desnecessárias
#axis=1 - elimina a coluna inteira
#inplace=True - atualiza o DataSet
base_treinamento.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
base_teste.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

#Transformando os atributos categóricos em dummies
new_base_treinamento = pd.get_dummies(base_treinamento)
new_base_teste = pd.get_dummies(base_teste)

#Comando para verificar se há valores nulos em alguma coluna. Pode ficar comentado depois.
#null_treino = new_base_treino.isnull().sum().sort_values(ascending=False)
#null_teste = new_base_teste.isnull().sum().sort_values(ascending=False)

new_base_treinamento['Age'].fillna(new_base_treinamento['Age'].mean(), inplace=True)
new_base_teste['Age'].fillna(new_base_teste['Age'].mean(), inplace=True)
new_base_teste['Fare'].fillna(new_base_teste['Fare'].mean(), inplace=True)

#Dividir os atributos em previsores e classe (classe é o que eu quero prever)
previsores = new_base_treinamento.drop('Survived', axis=1)
classe = new_base_treinamento['Survived']

#Fazer a divisão entre dados de treino e teste
from sklearn.model_selection import train_test_split

#previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#Naive Bayes: 79,85%
#from sklearn.naive_bayes import GaussianNB
#classificador = GaussianNB()
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#Árvore de Decisão: 82,09%
#from sklearn.tree import DecisionTreeClassifier

#classificador = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#Random Forest: 86,57%
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classificador.fit(previsores, classe)
previsoes = classificador.predict(new_base_teste)

#kNN: 64,92%
#from sklearn.neighbors import KNeighborsClassifier
#classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#Regressão Logística: 79,85%
#from sklearn.linear_model import LogisticRegression
#classificador = LogisticRegression()
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#SVM: 77,61%
#from sklearn.svm import SVC
#classificador = SVC(kernel = 'linear', random_state = 1)
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#AdaBoost: 81,34%
#from sklearn.ensemble import AdaBoostClassifier
#classificador = AdaBoostClassifier(n_estimators=200,random_state=0)
#classificador.fit(previsores_treinamento, classe_treinamento)
#previsoes = classificador.predict(previsores_teste)

#from sklearn.metrics import confusion_matrix, accuracy_score
#precisao = accuracy_score(classe_teste, previsoes)
#matriz = confusion_matrix(classe_teste, previsoes)

arquivo_final = pd.DataFrame()
arquivo_final['PassengerId'] = new_base_teste['PassengerId']
arquivo_final['Survived'] = previsoes
arquivo_final.to_csv('submission.csv', index=False)

print('Fim')

