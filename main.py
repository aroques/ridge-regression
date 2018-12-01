import numpy as np
import matplotlib.pyplot as plt

def main():
    x1 = np.random.uniform(-2, 10, 12)
    x_train = get_x_train(x1)
    y_train = get_y_train(x1)

    print_training_data(x1, y_train)
    
    w = np.dot(np.linalg.pinv(x_train), y_train)

    print_regression_eq(w)


def print_training_data(x1, y_train):
    print('(x, y) pairs:')
    print(str(np.column_stack((x1, y_train))) + '\n')


def print_regression_eq(w):
    print('y = {0}x + {1}'.format(round(w[1], 2), round(w[0], 2)))


def get_x_train(x1):
    bias = np.ones_like(x1)
    return np.column_stack((bias, x1))


def get_y_train(x1):
    return np.square(x1) + 10


if __name__ == '__main__':
    main()
 