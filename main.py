import numpy as np
import matplotlib.pyplot as plt

def main():
    x1 = np.random.uniform(low=-2, high=10, size=12)
    x_train = get_x_train(x1)
    y_train = get_y_train(x1)

    print_training_data(x1, y_train)

    w = linear_regression(x_train, y_train)
    best_r = determine_reg_param(x_train, y_train)
    w_reg = ridge_regression(x_train, y_train, regularizer=best_r) 
    
    print_regression_eq('without reg', w, round(calculate_error(w, x_train, y_train), 2))
    print()
    print_regression_eq('with reg', w_reg, round(calculate_error(w_reg, x_train, y_train), 2))
    
    plot_exp(x1, y_train, w, w_reg)


def determine_reg_param(x_train, y_train):
    regularizers = [0.1, 1, 10, 100]
    cv_error = np.zeros_like(regularizers)
    for i, r in enumerate(regularizers):
        cv_error[i] = cross_validation(x_train, y_train, 3, r)
    return regularizers[np.argmin(cv_error)]


def cross_validation(x_train, y_train, num_splits, regularizer):
    d = np.column_stack((x_train, y_train))
    cv_error = np.zeros(num_splits)
    
    for i, this_d in enumerate(np.split(d, num_splits)):
        x_train, y_train = this_d[:, :-1], this_d[:, -1] 
        w = ridge_regression(x_train, y_train, regularizer)
        cv_error[i] = calculate_error(w, x_train, y_train)

    return np.average(cv_error)


def calculate_error(w, x_train, y_train):
    y = np.dot(x_train, w)
    return np.average(np.square(np.subtract(y, y_train)))


def ridge_regression(x_train, y_train, regularizer):
    I = np.identity(x_train.shape[1])
    xTx = np.dot(x_train.T, x_train)
    xTy = np.dot(x_train.T, y_train)
    return np.dot(np.linalg.pinv(np.add(xTx, regularizer * I)), xTy)


def linear_regression(x_train, y_train):
    return np.dot(np.linalg.pinv(x_train), y_train)


def get_x_train(x1):
    bias = np.ones_like(x1)
    return np.column_stack((bias, x1))


def get_y_train(x1):
    return np.square(x1) + 10


def print_training_data(x1, y_train):
    print('(x, y) pairs:')
    print(str(np.column_stack((x1, y_train))) + '\n')


def print_regression_eq(label, w, error):
    print('{0:12}: y = {1}x + {2}'.format(label, round(w[1], 2), round(w[0], 2)))
    print('{0:12}: {1}'.format('error', error))


def plot_exp(x1, y_train, w, w_reg):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    
    ax.set(title='Ridge Regression')
    ax.scatter(x1, y_train, color='b', marker='x', label='training data')
    ax.plot(x1, line(w[1], x1, w[0]), color='g', label='regression line')
    ax.plot(x1, line(w_reg[1], x1, w_reg[0]), color='r', label='ridge regression line')
    
    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)
    plt.show()


def line(m, x, b):
    return m * x + b


if __name__ == '__main__':
    main()
 