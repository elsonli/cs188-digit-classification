import numpy as np

def main():
    """
    This is sample code for linear regression, which demonstrates how to use the
    Graph class.

    Once you have answered Questions 2 and 3, you can run `python nn.py` to
    execute this code.
    """

    # This is our data, where x is a 4x2 matrix and y is a 4x1 matrix
    x = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.dot(x, np.array([[7.],
                            [8.]])) + 3

    # Let's construct a simple model to approximate a function from 2D
    # points to numbers, f(x) = x_0 * m_0 + x_1 * m_1 + b
    # Here m and b are variables (trainable parameters):
    m = Variable(2,1)
    b = Variable(1)

    # We train our network using batch gradient descent on our data
    for iteration in range(10000):
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = Graph([m, b])
        input_x = Input(graph, x)
        input_y = Input(graph, y)
        xm = MatrixMultiply(graph, input_x, m)
        xm_plus_b = MatrixVectorAdd(graph, xm, b)
        loss = SquareLoss(graph, xm_plus_b, input_y)
        # Then we use the graph to perform backprop and update our variables
        graph.backprop()
        graph.step(0.01)

    # After training, we should have recovered m=[[7],[8]] and b=[3]
    print("Final values are: {}".format([m.data[0,0], m.data[1,0], b.data[0]]))
    assert np.isclose(m.data[0,0], 7)
    assert np.isclose(m.data[1,0], 8)
    assert np.isclose(b.data[0], 3)
    print("Success!")

class Graph(object):
    """
    A graph that keeps track of the computations performed by a neural network
    in order to implement back-propagation.

    Each evaluation of the neural network (during both training and test-time)
    will create a new Graph. The computation will add nodes to the graph, where
    each node is either a DataNode or a FunctionNode.

    A DataNode represents a trainable parameter or an input to the computation.
    A FunctionNode represents doing a computation based on two previous nodes in
    the graph.

    The Graph is responsible for keeping track of all nodes and the order they
    are added to the graph, for computing gradients using back-propagation, and
    for performing updates to the trainable parameters.

    For an example of how the Graph can be used, see the function `main` above.
    """

    def __init__(self, variables):
        """
        Initializes a new computation graph.

        variables: a list of Variable objects that store the trainable parameters
            for the neural network.

        Hint: each Variable is also a node that needs to be added to the graph,
        so don't forget to call `self.add` on each of the variables.
        """
        "*** YOUR CODE HERE ***"
        # Nodes need to be returned in order, so a list is better than other data structures.
        self.nodes = []
        self.variables = []
        self.node_vals = {}
        self.backprop_call = False
        for variable in variables:
            self.add(variable)
            self.variables.append(variable)

    def get_nodes(self):
        """
        Returns a list of all nodes that have been added to this Graph, in the
        order they were added. This list should include all of the Variable
        nodes that were passed to `Graph.__init__`.

        Returns: a list of nodes
        """
        "*** YOUR CODE HERE ***"
        return self.nodes

    def get_inputs(self, node):
        """
        Retrieves the inputs to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a list of numpy arrays

        Hint: every node has a `.get_parents()` method
        """
        "*** YOUR CODE HERE ***"
        curr_parents = node.get_parents()
        return [self.get_output(parent) for parent in curr_parents]

    def get_output(self, node):
        """
        Retrieves the output to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a numpy array or a scalar
        """
        "*** YOUR CODE HERE ***"
        return self.node_vals[node][0]

    def get_gradient(self, node):
        """
        Retrieves the gradient for a node in the graph. Assume the `node` has
        already been added to the graph.

        If `Graph.backprop` has already been called, this should return the
        gradient of the loss with respect to the output of the node. If
        `Graph.backprop` has not been called, it should instead return a numpy
        array with correct shape to hold the gradient, but with all entries set
        to zero.

        Returns: a numpy array
        """
        "*** YOUR CODE HERE ***"
        # If backprop has already been called, then return the gradient accumulator.
        if self.backprop_call:
            return self.node_vals[node][1]

        # Otherwise, return an array of zeros with the correct shape to hold the gradient.
        else:
            return np.zeros_like(self.get_output(node))

    def add(self, node):
        """
        Adds a node to the graph.

        This method should calculate and remember the output of the node in the
        forwards pass (which can later be retrieved by calling `get_output`)
        We compute the output here because we only want to compute it once,
        whereas we may wish to call `get_output` multiple times.

        Additionally, this method should initialize an all-zero gradient
        accumulator for the node, with correct shape.
        """
        "*** YOUR CODE HERE ***"
        # First, we need to add the node to our list of nodes.
        self.nodes.append(node)

        # We want to calculate the forward value and store this in a dictionary, so that
        # we only need to compute it once: memoization. Use the node as the dictionary key.
        inputs = self.get_inputs(node)
        forward_val = node.forward(inputs)

        # Initialize the all-zero gradient accumulator with the correct shape.
        gradient_accum = np.zeros_like(forward_val)

        # Update the node's value in the dictionary to contain the forward value
        # and the gradient accumulator.
        self.node_vals[node] = [forward_val, gradient_accum]

    def backprop(self):
        """
        Runs back-propagation. Assume that the very last node added to the graph
        represents the loss.

        After back-propagation completes, `get_gradient(node)` should return the
        gradient of the loss with respect to the `node`.

        Hint: the gradient of the loss with respect to itself is 1.0, and
        back-propagation should process nodes in the exact opposite of the order
        in which they were added to the graph.
        """
        loss_node = self.get_nodes()[-1]
        assert np.asarray(self.get_output(loss_node)).ndim == 0

        "*** YOUR CODE HERE ***"
        # Backprop has been called, update the value that was initialized to False.
        self.backprop_call = True
        # From part of the hint, we should process nodes in the opposite order.
        # Use [::-1] to both flip the list as well as making a copy of it.
        reversed_nodes = self.get_nodes()[::-1]

        # From the other part of the hint, the gradient of the loss w.r.t. itself is 1.0,
        # so just update the last node's gradient accumulator to be all ones.
        last_node = reversed_nodes[0]
        self.node_vals[last_node][1] = np.zeros_like(self.node_vals[last_node][1]) + 1

        # Loop over the reversed node list and calculate the backward value, which
        # takes in the inputs and the node's gradient.
        for node in reversed_nodes:
            backward_val = node.backward(self.get_inputs(node), self.get_gradient(node))

            # Now we want to pass this backward value to the parents.
            # We need to update the parent's gradient accumulator with the values calculated
            # in backward_val, and since they are in order, we can just loop over the range
            # and index appropriately.
            parent_nodes = node.get_parents()
            for index in range(len(parent_nodes)):
                self.node_vals[parent_nodes[index]][1] += backward_val[index]


    def step(self, step_size):
        """
        Updates the values of all variables based on computed gradients.
        Assume that `backprop()` has already been called, and that gradients
        have already been computed.

        Hint: each Variable has a `.data` attribute
        """
        "*** YOUR CODE HERE ***"
        # The variables in self.variables are the nodes, so we can use them to
        # index into the node dictionary we initialized in the beginning. Index
        # into the first position to get the gradient accumulator.
        for variable in self.variables:
            computed_grad = self.node_vals[variable][1]
            variable.data -= (step_size * computed_grad)

class DataNode(object):
    """
    DataNode is the parent class for Variable and Input nodes.

    Each DataNode must define a `.data` attribute, which represents the data
    stored at the node.
    """

    @staticmethod
    def get_parents():
        # A DataNode has no parent nodes, only a `.data` attribute
        return []

    def forward(self, inputs):
        # The forwards pass for a data node simply returns its data
        return self.data

    @staticmethod
    def backward(inputs, gradient):
        # A DataNode has no parents or inputs, so there are no gradients to
        # compute in the backwards pass
        return []

class Variable(DataNode):
    """
    A Variable stores parameters used in a neural network.

    Variables should be created once and then passed to all future Graph
    constructors. Use `.data` to access or modify the numpy array of parameters.
    """

    def __init__(self, *shape):
        """
        Initializes a Variable with a given shape.

        For example, Variable(5) will create 5-dimensional vector variable,
        while Variable(10, 10) will create a 10x10 matrix variable.

        The initial value of the variable before training starts can have a big
        effect on how long the network takes to train. The provided initializer
        works well across a wide range of applications.
        """
        assert shape
        limit = np.sqrt(3.0 / np.mean(shape))
        self.data = np.random.uniform(low=-limit, high=limit, size=shape)

class Input(DataNode):
    """
    An Input node packages a numpy array into a node in a computation graph.
    Use this node for inputs to your neural network.

    For trainable parameters, use Variable instead.
    """

    def __init__(self, graph, data):
        """
        Initializes a new Input and adds it to a graph.
        """
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.dtype.kind == "f", "data must have floating-point entries"
        self.data = data
        graph.add(self)

class FunctionNode(object):
    """
    A FunctionNode represents a value that is computed based on other nodes in
    the graph. Each function must implement both a forward and backward pass.
    """

    def __init__(self, graph, *parents):
        self.parents = parents
        graph.add(self)

    def get_parents(self):
        return self.parents

    @staticmethod
    def forward(inputs):
        raise NotImplementedError

    @staticmethod
    def backward(inputs, gradient):
        raise NotImplementedError

class Add(FunctionNode):
    """
    Adds two vectors or matrices, element-wise

    Inputs: [x, y]
        x may represent either a vector or a matrix
        y must have the same shape as x
    Output: x + y
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.add(np.array(inputs[0]), np.array(inputs[1]))

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # Assume f is the final node, as it is in our backpropagation problems.
        # Let A = input[0], B = input[1].
        # Let C = A + B. It will represent the output of A and B. It is the child function node of A and B.
        # df/dC = gradient (downstream gradient, given to us) 
        # 
        # Now, we want to be returning [df/input[0], df/input[1]].
        #
        # df/A = df/dC * dC/dA
        #
        # Let us find dC/dA.
        # dC/dA =  d(A + B)/dA = d(A+dA/dA + dB/dA = 1 + B(d(0)/dA) = 1
        # But, since everything here is a vector or matrix, this 1 is actually a vector or matrix of all 1's.
        # 
        # With all of that in consideration, we have df/A = df/dC * 1 = df/dC.
        # 
        # Therefore, df/A = gradient and we return [gradient, gradient]

        return [gradient, gradient]

class MatrixMultiply(FunctionNode):
    """
    Represents matrix multiplication.

    Inputs: [A, B]
        A represents a matrix of shape (n x m)
        B represents a matrix of shape (m x k)
    Output: a matrix of shape (n x k)
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.dot(np.array(inputs[0]), np.array(inputs[1]))

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # See Checkpoint 6 for why this particular formula.
        return [np.dot(gradient, inputs[1].T), np.dot(inputs[0].T, gradient)]

class MatrixVectorAdd(FunctionNode):
    """
    Adds a vector to each row of a matrix.

    Inputs: [A, x]
        A represents a matrix of shape (n x m)
        x represents a vector (m)
    Output: a matrix of shape (n x m)
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.add(np.array(inputs[0]), np.array(inputs[1]))

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # We want to return d(A + x)/dA = dA/dA + dx/dA = 1 + x(0).
        # Therefore, the answer is [gradient, gradient] as well.
        #
        # However, with one caveat. The second gradient we return needs to
        # have the same size and shape of its input (the vector x).
        # As mentioned on Piazza, this means we need to aggregate the gradient
        # (because if the entire gradient has the shape of a matrix, that doesn't match)
        # and we can do this with np.sum(). 
        #
        # The axis here tells sum to combine the values in a vertical fashion.
        # e.g., let E = [0 1]
        #               [2 3]
        # np.sum(E, axis=0) means we'll get back sum_E = [2 4]
        return [gradient, np.sum(gradient, axis=0)]

class ReLU(FunctionNode):
    """
    An element-wise Rectified Linear Unit nonlinearity: max(x, 0).
    This nonlinearity replaces all negative entries in its input with zeros.

    Input: [x]
        x represents either a vector or matrix
    Output: same shape as x, with no negative entries
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.maximum(inputs[0], np.zeros_like(inputs[0]))

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"

        # The derivative of ReLU is 1 if x > 0 and 0 otherwise.
        # Define a function that does this for us locally (this does not work
        # if defined outside because it is a static method).
        def relu_zero_or_one(value):
            return 1 if value > 0 else 0

        # Transform this function into something that can operate on and
        # return numpy arrays with np.vectorize.
        relu_zero_or_one = np.vectorize(relu_zero_or_one)

        # Just return the product of this function's result with the local gradient.
        return [np.multiply(relu_zero_or_one(inputs[0]), gradient)]

class SquareLoss(FunctionNode):
    """
    Inputs: [a, b]
        a represents a matrix of size (batch_size x dim)
        b must have the same shape as a
    Output: a number

    This node first computes 0.5 * (a[i,j] - b[i,j])**2 at all positions (i,j)
    in the inputs, which creates a (batch_size x dim) matrix. It then calculates
    and returns the mean of all elements in this matrix.
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        return np.mean(0.5 * np.square(np.subtract(np.array(inputs[0]), np.array(inputs[1]))))

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # The derivative is just (a[i,j] - b[i,j]) at all positions (i,j).
        difference = np.subtract(np.array(inputs[0]), np.array(inputs[1]))

        # Calculate the mean and update all the elements of the matrix.
        difference /= difference.size

        # The test outputted [[9][-5]] and [[-9][5]] so it seems that I just
        # had to multiply this by -1??
        return [gradient * difference, -1 * gradient * difference]

class SoftmaxLoss(FunctionNode):
    """
    A batched softmax loss, used for classification problems.

    IMPORTANT: do not swap the order of the inputs to this node!

    Inputs: [logits, labels]
        logits: a (batch_size x num_classes) matrix of scores, that is typically
            calculated based on previous layers. Each score can be an arbitrary
            real number.
        labels: a (batch_size x num_classes) matrix that encodes the correct
            labels for the examples. All entries must be non-negative and the
            sum of values along each row should be 1.
    Output: a number

    We have provided the complete implementation for your convenience.
    """
    @staticmethod
    def softmax(input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def forward(inputs):
        softmax = SoftmaxLoss.softmax(inputs[0])
        labels = inputs[1]
        assert np.all(labels >= 0), \
            "Labels input to SoftmaxLoss must be non-negative. (Did you pass the inputs in the right order?)"
        assert np.allclose(np.sum(labels, axis=1), np.ones(labels.shape[0])), \
            "Labels input to SoftmaxLoss do not sum to 1 along each row. (Did you pass the inputs in the right order?)"

        return np.mean(-np.sum(labels * np.log(softmax), axis=1))

    @staticmethod
    def backward(inputs, gradient):
        softmax = SoftmaxLoss.softmax(inputs[0])
        return [
            gradient * (softmax - inputs[1]) / inputs[0].shape[0],
            gradient * (-np.log(softmax)) / inputs[0].shape[0]
        ]

if __name__ == '__main__':
    main()
