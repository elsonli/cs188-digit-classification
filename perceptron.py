import numpy as np

import backend

class Perceptron(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.get_data_and_monitor = backend.make_get_data_and_monitor_perceptron()

        # "*** YOUR CODE HERE ***"
        self.weight_vector = np.zeros(dimensions)

    def get_weights(self):
        """
        Return the current weights of the perceptron.

        Returns: a numpy array with D elements, where D is the value of the
            `dimensions` parameter passed to Perceptron.__init__
        """

        # "*** YOUR CODE HERE ***"
        return self.weight_vector

    def predict(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """

        # "*** YOUR CODE HERE ***"
        dot_product = np.dot(self.weight_vector, x)
        if dot_product >= 0:
            return 1
        else:
            return -1

    def update(self, x, y):
        """
        Update the weights of the perceptron based on a single example.
            x is a numpy array with D elements, where D is the value of the
                `dimensions`  parameter passed to Perceptron.__init__
            y is either 1 or -1

        Returns:
            True if the perceptron weights have changed, False otherwise
        """

        # "*** YOUR CODE HERE ***"
        if self.predict(x) == y:
            return False
        scaled_x = y * x
        self.weight_vector = self.weight_vector + scaled_x
        return True

    def train(self):
        """
        Train the perceptron until convergence.

        To iterate through all of the data points once (a single epoch), you can
        do:
            for x, y in self.get_data_and_monitor(self):
                ...

        get_data_and_monitor yields data points one at a time. It also takes the
        perceptron as an argument so that it can monitor performance and display
        graphics in between yielding data points.
        """
        "*** YOUR CODE HERE ***"

        converged = False
        while not converged:
            initial_weights = self.weight_vector
            for x, y in self.get_data_and_monitor(self):
                self.update(x,y)
            if np.array_equal(initial_weights, self.weight_vector):
                converged = True
