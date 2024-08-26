import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv('heart.csv')
print(df.head())


#Checking if there is null
print(df.isnull().sum())
data = df.dropna()

X = df[['age','trtbps','chol']]
y = df['output']

print(X.head())
print(y.head())
print(X.tail())
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X,y)

prediction = model.predict([[56,120,236]])
print(f'The prediction is :{prediction}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42 )
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train,y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

score = accuracy_score(y_train, y_pred_train)
print(f"Model Accuracy Train:{score:.2f}" )
score_test = accuracy_score(y_test, y_pred_test)
print(f"Model Accuracy Test:{score_test:.2f}")

plt.scatter(df['age'], df['chol'], color='blue')
plt.xlabel('Age')
plt.ylabel('Cholesterol (chol)')
plt.title('Age vs Cholesterol')
plt.show()