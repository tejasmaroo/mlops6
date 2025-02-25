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

# Set a fixed random seed for reproducibility
fixed_seed = 42

for i in range(10):  # Ensure colon `:` is present
    rng = np.random.RandomState(fixed_seed + i)  # Correctly indented inside loop
    
    x = 10 * rng.rand(1000).reshape(-1, 1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=fixed_seed)

    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)

    test_mse = mean_squared_error(y_test, y_preds)
    average_mse = np.mean(test_mse)

    print(f'MSE Result: {test_mse}')
    print(f'Average Mean Squared Error: {average_mse}')

    # Append results to the file instead of overwriting
    with open("metrics.txt", "a") as outfile:
        outfile.write(f'Mean Squared Error = {average_mse}\n')

    # Plot the results
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
