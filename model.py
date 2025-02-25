import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

lr = LinearRegression()

if not os.path.exists("metrics.txt"):
    open("metrics.txt", "w").close()  # Ensure file exists

for _ in range(10):  # ← Make sure the colon (:) is present
    rng = np.random.RandomState(_)  # ← This must be indented properly
    x = 10 * rng.rand(1000).reshape(-1, 1)
    y = 2 * x - 5 + rng.randn(1000).reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    
    lr.fit(X_train, y_train)
    y_preds = lr.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_preds)
    average_mse = np.mean(test_mse)
    
    print(f'MSE Result: {test_mse}')
    print("Average Mean Squared Error:", average_mse)
    
    with open("metrics.txt", "a") as outfile:  # Append instead of overwrite
        outfile.write(f'Mean Squared Error = {average_mse}\n')

    # Plotting
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='red', label='Testing data')
    plt.scatter(X_test, y_preds, c="g", label="Predictions")
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Training and Testing Data Split - Iteration {_}')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'model_results_{_}.png', dpi=120)
    plt.close()  # Prevent overlapping plots
