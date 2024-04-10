import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('TSLA.csv')
X = df[['High', 'Low', 'Open']]
y = df['Close']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump((model, scaler), 'model_v1.pkl')

if __name__ == "__main__":
    print("Modelo guardado")

