# Linear Regression from Scratch vs. Scikit-Learn vs. Ordinary Least Squares (OLS)
In this notebook, we will implement linear regression from scratch using numpy and compare it with the linear regression model from scikit-learn and the ordinary least squares method.

## Our Model

### Data Preparation

Because we are using gradient descent to optimize the weights of our model, we need to normalize the data. 

```python
def prep_data(X, y):
    # normalize x to have mean=0, std=1
    X = (X - X.mean(axis=0)) / X.std(axis=0) + 1

    # Normalize y to have mean=1, std=1
    y = (y - y.mean()) / y.std() + 1
    return X, y
```

### Fitting the Model

We use gradient descent to optimize the weights of our model. 

#### Loss Function

We use mean squared error as the loss function.

$$
\begin{equation}
L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (z_i - \hat{y}_i)^2
\end{equation}
$$

#### Optimization

To optimize the weights of our model, we must get the gradient of the loss function with respect to the weights.

$$
\begin{equation}
\frac{\partial L}{\partial W} = \frac{2}{N} X^T(XW - y)
\end{equation}
$$

$$
\begin{equation}
\frac{\partial L}{\partial b} = \frac{2}{N} (XW - y)
\end{equation}
$$


```python
def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            
            dw = (2 / n_samples) * np.dot(X.T, (z - y))
            db = (2 / n_samples) * np.sum(z - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
```

We update the weights and biases using the following formulas:

$$
\begin{equation}
W = W - \alpha \frac{\partial L}{\partial W}
\end{equation}
$$

Where, hyperparameter, $\alpha$ is the learning rate.

$$
\begin{equation}
b = b - \alpha \frac{\partial L}{\partial b}
\end{equation}
$$

### Scoring the Model

We use the coefficient of determination, $R^2$ to score our model.

$$
\begin{equation}
R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
\end{equation}
$$

```python
def score(self, X, y):  # R^2 score
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / (v + 1e-10)  # Add small constant to avoid division by zero
```

# Ordinary Least Squares (OLS)

Just like our model, we prep our data exactly the same. However, for the fit function, we use the following formula to calculate the weights:


$$
\begin{equation}
\theta = (X^T X)^{-1} X^T y
\end{equation}
$$

```python
def fit(self, X, y):
         # Add a column of ones to X for the bias term
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Compute the coefficients using the normal equation
        coeffs = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        # Extract bias and weights
        self.bias = coeffs[0] 
        self.weights = coeffs[1:]
```

We have to an add a column of ones to X for the bias term because our X only has features and no bias term. Then we use the normal equation to calculate the weights and bias. 

### Scoring the Model

We score this model with an $R^2$ score as well.

# Comparing Models

After implementing our linear regression model from scratch, the final $R^2$ score is effectively equivalent to the $R^2$ score from scikit-learn's linear regression model.

Scikit-learn's R^2 score: 0.49656835105076846

Our Model: 0.4965683510508899

OLS Model: 0.49656835105076835