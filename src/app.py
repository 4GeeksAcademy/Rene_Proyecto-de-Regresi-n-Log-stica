from utils import db_connect
engine = db_connect()

df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep=';')

# Verificamos columnas separadas correctamente
print(df.columns.tolist())

print(df.head(5))

df.info()

df.isnull().sum()


sns.countplot(data=df, x='y')
plt.title('Distribución del Target (Depósito contratado)')
plt.show()

print(df['y'].value_counts(normalize=True) * 100)



plt.Figure(figsize=(10,5))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Distribucion de la edad')
plt.xlabel('Edad')
plt.show()


plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='job', hue='y')
plt.title("Tipo de trabajo vs. Depósito contratado")
plt.xticks(rotation=45)
plt.show()



cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'day_of_week', 'poutcome']

for var in cat_vars:
    print(f"\n{var}:\n", df[var].value_counts())



plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Mapa de calor de correlaciones")
plt.show()


sns.countplot(data=df, x='y')
plt.title('Distribución del Target (Depósito contratado)')
plt.show()

print(df['y'].value_counts(normalize=True) * 100)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


df['y'] = df['y'].map({'yes': 1, 'no': 0})
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


model_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
model_balanced.fit(X_train_scaled, y_train)

y_pred_bal = model_balanced.predict(X_test_scaled)

print(confusion_matrix(y_test, y_pred_bal))
print(classification_report(y_test, y_pred_bal))


pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

# Aplicar SMOTE al set de entrenamiento
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

# Entrenar nuevo modelo
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_resampled, y_resampled)

y_pred_smote = model_smote.predict(X_test_scaled)

print(confusion_matrix(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))


