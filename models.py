import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        # First tried a learning rate of 0.10, and although the error seemed to be
        # fairly low, it was not passing the tests. Trying 0.05, the learning rate
        # from the graph seemed to be fairly reasonable as it slowed down towards
        # later iterations. With 300 hidden layers, it caused the performance to
        # drop drastically, so I lowered it down to 50, which still failed the tests,
        # and then I went back to 100 which passed.
        self.learning_rate = 0.05
        self.hidden_layers = 100

        # Create variables for the ones present in the following equation:
        # f(x) = W2 * ReLU(W1 * x + b1) + b2
        # Make it so that the dimensions of the variables match such that matrix
        # operations would be legal/valid.
        self.W1 = nn.Variable(1, self.hidden_layers)
        self.b1 = nn.Variable(self.hidden_layers)
        self.W2 = nn.Variable(self.hidden_layers, 1)
        self.b2 = nn.Variable(1)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # Implemented based on the equation posted by Yichi Zhang on Piazza:
        # f(x) = W2 * ReLU(W1 * x + b1) + b2
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, x)
        W1_x = nn.MatrixMultiply(graph, input_x, self.W1)
        W1_x_plus_b1 = nn.MatrixVectorAdd(graph, W1_x, self.b1)
        relu = nn.ReLU(graph, W1_x_plus_b1)
        W2_relu = nn.MatrixMultiply(graph, relu, self.W2)
        W2_relu_plus_b2 = nn.MatrixVectorAdd(graph, W2_relu, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            W2_relu_plus_b2_loss = nn.SquareLoss(graph, W2_relu_plus_b2, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(W2_relu_plus_b2)


class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        # Start off with the learning rate and hidden layers from the previous
        # problem, then adjust if the tests fail.
        self.learning_rate = 0.05
        self.hidden_layers = 150

        # Start off with the same parameters as the previous problem.
        self.W1 = nn.Variable(1, self.hidden_layers)
        self.b1 = nn.Variable(self.hidden_layers)
        self.W2 = nn.Variable(self.hidden_layers, 1)
        self.b2 = nn.Variable(1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # Implemented based on the equation posted by Zhuofan Zhang on Piazza:
        # f(x) = W2 * ReLU(W1 * x + b1) - ReLU(W1 * (-x) + b1)

        # First, calculate the W2 * ReLU(W1 * x + b1) portion.
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, x)
        W1_x = nn.MatrixMultiply(graph, input_x, self.W1)
        W1_x_plus_b1 = nn.MatrixVectorAdd(graph, W1_x, self.b1)
        relu1 = nn.ReLU(graph, W1_x_plus_b1)
        W2_relu1 = nn.MatrixMultiply(graph, relu1, self.W2)
        W2_relu1_plus_b2 = nn.MatrixVectorAdd(graph, W2_relu1, self.b2)

        # Next, calculate the ReLU(W1 * (-x) + b1) portion. The code is basically
        # the same as the calculated portion above.
        neg_x = nn.Input(graph, -1 * x)
        W1_neg_x = nn.MatrixMultiply(graph, neg_x, self.W1)
        W1_neg_x_plus_b1 = nn.MatrixVectorAdd(graph, W1_neg_x, self.b1)
        relu2 = nn.ReLU(graph, W1_neg_x_plus_b1)
        W2_relu2 = nn.MatrixMultiply(graph, relu2, self.W2)
        W2_relu2_plus_b2 = nn.MatrixVectorAdd(graph, W2_relu2, self.b2)

        # We don't have a nn.Subtract function so we need to multiply
        # W2_relu2_plus_b2 by -1 so we can just use nn.Add on the two.
        neg_W2_relu2_plus_b2 = nn.Input(graph, -1 * graph.get_output(W2_relu2_plus_b2))
        final_matrix = nn.Add(graph, W2_relu1_plus_b2, neg_W2_relu2_plus_b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            final_matrix_loss = nn.SquareLoss(graph, final_matrix, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(final_matrix)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        # Start off with the learning rate and hidden layers from the previous
        # problem, then adjust if the tests fail.
        self.learning_rate = 0.2
        self.hidden_layers = 200

        # Start off with the same parameters as the previous problem.
        # Change self.W1 to contain 784 because that is the size of the dimensional vector.
        # Change self.W2 and self.b2 to contain 10 because that is the size of the output.
        self.W1 = nn.Variable(784, self.hidden_layers)
        self.b1 = nn.Variable(self.hidden_layers)
        self.W2 = nn.Variable(self.hidden_layers, 10)
        self.b2 = nn.Variable(10)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        # Implemented based on the equation posted by Yichi Zhang on Piazza:
        # f(x) = W2 * ReLU(W1 * x + b1) + b2
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, x)
        W1_x = nn.MatrixMultiply(graph, input_x, self.W1)
        W1_x_plus_b1 = nn.MatrixVectorAdd(graph, W1_x, self.b1)
        relu = nn.ReLU(graph, W1_x_plus_b1)
        W2_relu = nn.MatrixMultiply(graph, relu, self.W2)
        W2_relu_plus_b2 = nn.MatrixVectorAdd(graph, W2_relu, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            W2_relu_plus_b2_loss = nn.SoftmaxLoss(graph, W2_relu_plus_b2, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(W2_relu_plus_b2)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        self.hidden_layers = 100

        # We are going to be implementing the Q function mentioned in the spec 
        # which takes the form of :
        # Q(s,a) = w.T * f(s,a) = w0f0(s,a) + ... + wnfn(s,a)
        # The following parameters are for defining this Q(s,a) we are replacing.
        self.W1 = nn.Variable(4, self.hidden_layers)
        self.b1 = nn.Variable(self.hidden_layers)
        self.W2 = nn.Variable(self.hidden_layers, 2)
        self.b2 = nn.Variable(2)
    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_states = nn.Input(graph, states)
        W1_states = nn.MatrixMultiply(graph, input_states, self.W1)
        W1_states_plus_b1 = nn.MatrixVectorAdd(graph, W1_states, self.b1)
        relu = nn.ReLU(graph, W1_states_plus_b1)
        W2_relu = nn.MatrixMultiply(graph, relu, self.W2)
        W2_relu_plus_b2 = nn.MatrixVectorAdd(graph, W2_relu, self.b2)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            input_q_target = nn.Input(graph, Q_target)
            W2_relu_plus_b2_loss = nn.SquareLoss(graph, input_q_target, W2_relu_plus_b2)
            return graph
        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(W2_relu_plus_b2)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        self.hidden_layers = 300

        self.W1 = nn.Variable(47, self.hidden_layers)
        self.b1 = nn.Variable(self.hidden_layers)
        self.W2 = nn.Variable(self.hidden_layers, 5)
        self.b2 = nn.Variable(5)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"
        # Initiate graph outside of the for loop so we can use that to initiate input_h.
        # Initiate input_h outside the for loop so that it can be iteratively updated.
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_h = nn.Input(graph, np.zeros((batch_size, self.hidden_layers)))

        # Use basically the same equation as above but from anonymous post on Piazza,
        # use the nn.Add() to add h and c.
        for character in xs:
            input_character = nn.Input(graph, character)
            W1_character = nn.MatrixMultiply(graph, input_character, self.W1)
            W1_character_plus_b1 = nn.MatrixVectorAdd(graph, W1_character, self.b1)
            W1_character_plus_b1_plus_h = nn.Add(graph, W1_character_plus_b1, input_h)
            W1_relu = nn.ReLU(graph, W1_character_plus_b1_plus_h)
            input_h = W1_relu
            W2_relu = nn.MatrixMultiply(graph, W1_relu, self.W2)
            W2_relu_plus_b2 = nn.MatrixVectorAdd(graph, W2_relu, self.b2)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            W2_relu_plus_b2_loss = nn.SoftmaxLoss(graph, W2_relu_plus_b2, input_y)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(W2_relu_plus_b2)
