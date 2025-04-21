import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
# the Regression that'll be used to train the model on the given dataset (mention this in the README)
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# load dataset (again? lol)
df = pd.read_csv('data/winequality-red.csv', delimiter=';')

# set the given features and targets
X = df.drop('quality', axis=1) # first matrix
y = df['quality']

# scaling the given features (this makes it easier for the model to differentiate and not be racist)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test split --- 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # same split (?)

# train model finally
model = RandomForestRegressor()
model.fit(X_train, y_train)

# prediction and eval
y_pred = model.predict(X_test)
print('r2 score:', r2_score(y_test, y_pred))
print("mse:", mean_squared_error(y_test, y_pred))

# scale the model and train it (save)
joblib.dump(model, 'wine_model.pkl')
joblib.dump(scaler, 'scaler.pkl')