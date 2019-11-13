import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import operator

sns.set()

np.random.seed(0)


def linear_regression(x,y):
    # Linear model
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    y_bias = (x - np.mean(y_pred, axis=1)) ** 2
    y_var = explained_variance_score(y, y_pred)

    print("R2 error {}".format(model.score(x, y)))
    print("RSME error {}".format(np.sqrt(mean_squared_error(y, y_pred))))

    print("Variance {}".format(np.mean(y_var)))

    plt.scatter(x, y, s=10)
    plt.plot(x, y_pred, color='r', label='linear regression')


def polynomial(x, y, degree, color):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    plt.scatter(x, y, s=10)

    y_bias = (x - np.mean(y_poly_pred, axis=1)) ** 2
    y_var = explained_variance_score(y, y_poly_pred)

    print("R2 error {}".format(r2_score(y, y_poly_pred)))
    print("RSME error {}".format(np.sqrt(mean_squared_error(y, y_poly_pred))))

    print("Variance {}".format(np.mean(y_var)))

    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color=color, label='Degree {}'.format(degree))


def main():


    x = 2 - 3 * np.random.normal(0, 1, 20)
    y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

    # transforming the data to include another axis
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]


    print("Linear Regession model")
    linear_regression(x,y)

    print()
    # Polynomial fitting
    print("Polynomial model degree 2")
    polynomial(x, y, 2, "g")
    plt.legend( loc='upper left' )
    plt.show()


    print()
    # Polynomial fitting
    print("Polynomial model degree 4")
    polynomial(x, y, 15, "m")

    plt.legend( loc='upper left' )
    plt.show()


if __name__ == '__main__':
    main()