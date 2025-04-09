# Predicting housing prices with neural networks
# Packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU
import matplotlib.pyplot as plt
import pandas as pd
# Set plot style
plt.style.use('seaborn-v0_8')
# Loading and Reprocessing
# Generate Synthetic data

#np.random.seed(42)
#n_samples = 1000
#X = np.random.rand(n_samples, 3) * 100 # Features: size, location, bedrooms
#y = X[:, 0] * 3 + X[:, 1] * 2 + X[:, 2] * 4 + np.random.rand(n_samples) * 10


# Real data using Pandas
df = pd.read_csv('housing1.csv')
df.head()

X = df[['size', 'location_score', 'bedrooms']]
y = df['price']
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Neural Network Design
# Define the model
model = Sequential([
    Dense(64, input_shape=(3,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1) # Output layer for regression
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), # try a differnt learning rate
    loss='mean_squared_error',
    metrics=['mae'] # mean absoulute error
)
# Training the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32
)
# Evaluating the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')
# Predictions
predictions = model.predict(X_test)
# Visualization - True vs predicted prices
plt.scatter(y_test, predictions)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True Prices vs Predicted Prices')
plt.show()
