#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
#                       Computational Graphs
# =============================================================================
class Operation:
    """Represents a graph node that performs a computation.

    An `Operation` is a node in a `Graph` that takes zero or
    more objects as input, and produces zero or more objects
    as output.
    """

    def __init__(self, input_nodes=[]):
        """Construct Operation
        """
        self.input_nodes = input_nodes

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def compute(self):
        """Computes the output of this operation.
        "" Must be implemented by the particular operation.
        """
        pass

class add(Operation):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        return x_value + y_value

class matmul(Operation):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        return a_value.dot(b_value)

class placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
        self.consumers = []

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)

class Variable:
    """Represents a variable (i.e. an intrinsic, changeable parameter of a computational graph).
    """

    def __init__(self, initial_value=None):
        """Construct Variable

        Args:
          initial_value: The initial value of this variable
        """
        self.value = initial_value
        self.consumers = []

        # Append this variable to the list of variables in the currently active default graph
        _default_graph.variables.append(self)
class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self


# Create a new graph
Graph().as_default()

# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# Create placeholder
x = placeholder()

# Create hidden node y
y = matmul(A, x)

# Create output node z
z = add(y, b)

import numpy as np


class Session:
    """Represents a particular execution of a computational graph.
    """

    def run(self, operation, feed_dict={}):
        """Computes the output of an operation

        Args:
          operation: The operation whose output we'd like to compute.
          feed_dict: A dictionary that maps placeholders to values for this session
        """

        # Perform a post-order traversal of the graph to bring the nodes into the right order
        nodes_postorder = traverse_postorder(operation)

        # Iterate all nodes to determine their value
        for node in nodes_postorder:

            if type(node) == placeholder:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == Variable:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else:  # Operation
                # Get the input values for this operation from the output values of the input nodes
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        return operation.output


def traverse_postorder(operation):
    """Performs a post-order traversal, returning a list of nodes
    in the order in which they have to be computed

    Args:
       operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder

# Execution
session = Session()
output = session.run(z, {
    x: [1, 2]
})
print(output)

# =============================================================================
#                   Perceptron
# =============================================================================

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2*np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2*np.ones((50, 2))

import matplotlib.pyplot as plt
# Plot the red and blue points
plt.scatter(red_points[:,0], red_points[:,1], color='red')
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue')

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)

# Create an interval from -5 to 5 in steps of 0.01
a = np.arange(-5, 5, 0.01)

# Compute corresponding sigmoid function values
s = 1 / (1 + np.exp(-a))

# Plot them
plt.plot(a, s)
plt.grid(True)
plt.show()

class sigmoid(Operation):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))
    
# Create a new graph
Graph().as_default()

x = placeholder()
w = Variable([1, 1])
b = Variable(0)
p = sigmoid( add(matmul(w, x), b) )

"""
Let's use this perceptron to compute the probability that 
(3,2)T is a blue point:
"""
session = Session()
print(session.run(p, {
    x: [3, 2]
}))

# =============================================================================
#                   Multi-class perceptron
# =============================================================================
class softmax(Operation):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        """Construct softmax

        Args:
          a: Input node
        """
        super().__init__([a])

    def compute(self, a_value):
        """Compute the output of the softmax operation

        Args:
          a_value: Input value
        """
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]

# Create a new graph
Graph().as_default()

X = placeholder()

# Create a weight matrix for 2 output classes:
# One with a weight vector (1, 1) for blue and one with a weight vector (-1, -1) for red
W = Variable([
    [1, -1],
    [1, -1]
])
b = Variable([0, 0])
p = softmax( add(matmul(X, W), b) )

# Create a session and run the perceptron on our blue/red points
session = Session()
output_probabilities = session.run(p, {
    X: np.concatenate((blue_points, red_points))
})

# Print the first 10 lines, corresponding to the probabilities of the first 10 points
print(output_probabilities[:10])


# =============================================================================
#                       Training criterion
# =============================================================================

""" This computes the element-wise logarithm of a tensor."""
class log(Operation):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x):
        """Construct log

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return np.log(x_value)

""" This computes the element-wise product of two tensors of the same shape."""
class multiply(Operation):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """
        return x_value * y_value

class reduce_sum(Operation):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super().__init__([A])
        self.axis = axis

    def compute(self, A_value):
        """Compute the output of the reduce_sum operation

        Args:
          A_value: Input tensor value
        """
        return np.sum(A_value, self.axis)

""" This computes the element-wise negative of a tensor."""
class negative(Operation):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        """Construct negative

        Args:
          x: Input node
        """
        super().__init__([x])

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value

c = placeholder()
J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

""" Let's now compute the loss of our red/blue perceptron."""
# Create a new graph
Graph().as_default()

X = placeholder()
c = placeholder()

W = Variable([
    [1, -1],
    [1, -1]
])
b = Variable([0, 0])
p = softmax(add(matmul(X, W), b))

# Cross-entropy loss
J = negative(reduce_sum(reduce_sum(multiply(c, log(p)), axis=1)))

session = Session()
print(session.run(J, {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}))