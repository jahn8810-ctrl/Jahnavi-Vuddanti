import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("housing.csv")       # Your file here
df = df.drop(['Id'], axis=1, errors='ignore')


X = df.drop('Price', axis=1).values
y = df['Price'].values.reshape(-1,1)
m, n = X.shape


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled = np.hstack([np.ones((m,1)), X_scaled])

# ----------------------------------------------------
# 3. Cost Function
# ----------------------------------------------------
def compute_cost(X, y, w):
    m = len(y)
    return (1/(2*m)) * np.sum((X.dot(w) - y) ** 2)

# ----------------------------------------------------
# 4. Gradient Function
# ----------------------------------------------------
def compute_gradient(X, y, w):
    m = len(y)
    error = X.dot(w) - y
    gradient = (1/m) * (X.T.dot(error))     # shape = (n+1, 1)
    return gradient

# ----------------------------------------------------
# 5. Gradient Descent
# ----------------------------------------------------
def gradient_descent(X, y, w, alpha, iters):
    J_history = []
    for i in range(iters):
        grad = compute_gradient(X, y, w)
        w = w - alpha * grad
        J_history.append(compute_cost(X, y, w))
    return w, J_history

# ----------------------------------------------------
# 6. Train Model
# ----------------------------------------------------
w = np.zeros((n+1, 1))   # w0 = bias, w1..wn = weights
alpha = 0.01
iters = 500

w_final, cost = gradient_descent(X_scaled, y, w, alpha, iters)
print("Weights Learned:\n", w_final)

# ----------------------------------------------------
# 7. Plot Cost History
# ----------------------------------------------------
plt.plot(cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()

# ----------------------------------------------------
# 8. UNIVERSAL PREDICTION FUNCTION
# ----------------------------------------------------
def predict(input_features):
    """
    input_features : list or array of feature values [x1, x2, x3, ...]
    """
    input_features = np.array(input_features).reshape(1, -1)

    # scale using SAME scaler
    scaled = scaler.transform(input_features)

    # add bias
    scaled = np.hstack([np.ones((1,1)), scaled])

    return scaled.dot(w_final)

# ----------------------------------------------------
# 9. Try prediction (works with ANY number of features)
# ----------------------------------------------------
example = [1650, 3]     # give ANY length based on your dataset
print("Predicted Price:", predict(example))
