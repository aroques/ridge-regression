import numpy as np
import matplotlib.pyplot as plt

def main():
    x1 = np.random.uniform(low=-2, high=10, size=12)
    x_train = get_x_train(x1)
    y_train = get_y_train(x1)

    print_training_data(x1, y_train)

    w = np.dot(np.linalg.pinv(x_train), y_train)

    print_regression_eq(w)
    
    plot_exp(x1, y_train, w)


def get_x_train(x1):
    bias = np.ones_like(x1)
    return np.column_stack((bias, x1))


def get_y_train(x1):
    return np.square(x1) + 10


def print_training_data(x1, y_train):
    print('(x, y) pairs:')
    print(str(np.column_stack((x1, y_train))) + '\n')


def print_regression_eq(w):
    print('y = {0}x + {1}'.format(round(w[1], 2), round(w[0], 2)))


def plot_exp(x1, y_train, w):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    
    ax.set(title='Ridge Regression')
    ax.scatter(x1, y_train, color='b', marker='x', label='training data')
    ax.plot(x1, line(w[1], x1, w[0]), color='g', label='regression line')
    
    ax.legend(facecolor='w', fancybox=True, frameon=True, edgecolor='black', borderpad=1)
    plt.show()


def line(m, x, b):
    return m * x + b


if __name__ == '__main__':
    main()
 