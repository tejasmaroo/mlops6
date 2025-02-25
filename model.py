import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

lr = LinearRegression()

# Ensure metrics file exists before writing
if not os.path.exists("metrics.txt"):
    open("metrics.txt", "w").close()

mse_results = []

for _ in range(10):
    rng = np.random.RandomState(42)  # Fixed random seed for reproducibility
    
    x = 10 * rng.rand(1000).reshape(-1, 1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)

    test_mse = mean_squared_error(y_test, y_preds)
    mse_results.append(test_mse)

    print(f'MSE Result: {test_mse}')

# Calculate the average MSE after the loop
average_mse = np.mean(mse_results)
print(f'Average Mean Squared Error: {average_mse}')

# Append results to the file instead of overwriting
with open("metrics.txt", "a") as outfile:
    outfile.write(f'Average Mean Squared Error = {average_mse}\n')

# Plot the results
for i, (X_train, y_train, X_test, y_test, y_preds) in enumerate(zip(X_train, y_train, X_test, y_test, y_preds)):
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='red', label='Testing data')
    plt.scatter(X_test, y_preds, color="green", label="Predictions")

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Training and Testing Data Split - Iteration {i}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'model_results_{i}.png', dpi=120)
    plt.close()  # Prevents overlapping plots
