from builtins import range
from builtins import object
import numpy as np

from libs.layers import *
from libs.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        output_1, cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
        output_2, cache2 = relu_forward(output_1)
        scores, cache3 = affine_forward(output_2, self.params['W2'], self.params['b2'])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss1, out_1 = softmax_loss(scores, y)
        loss2 = 0.5 * self.reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))
        loss = loss1 + loss2
        k1, grads['W2'], grads['b2'] = affine_backward(out_1, cache3)
        k2 = relu_backward(k1, cache2)
        unused, grads['W1'], grads['b1'] = affine_backward(k2, cache1)
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout as option. For a network with L layers,
    the architecture will be

    {affine - relu - [dropout]} x (L - 1) - affine - softmax

    where dropout is optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        if self.num_layers >= 3:
            for i in range(2, self.num_layers):
                self.params['W' + str(i)] = np.random.normal(scale=weight_scale, size=(hidden_dims[i-2], hidden_dims[i-1]))
                self.params['b' + str(i)] = np.zeros(hidden_dims[i-1])        
        self.params['W' + str(self.num_layers)] = np.random.normal(scale=weight_scale, size=(hidden_dims[-1], num_classes))
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dic1, dic2 = {}, {}
        dic2['out1'], dic1['out1'] = affine_forward(X, self.params['W1'], self.params['b1'])
        dic2['relu_1'], dic1['relu_1'] = relu_forward(dic2['out1'])
        if self.use_dropout:
            dic2['dropout_1'], dic1['dropout_1'] = dropout_forward(dic2['relu_1'], self.dropout_param)
        if self.num_layers >= 3:
              for i in range(2, self.num_layers):
                if self.use_dropout:
                    dic2['out' + str(i)], dic1['out' + str(i)] = affine_forward(dic2['dropout_' + str(i-1)], self.params['W' + str(i)], self.params['b' + str(i)])
                else:
                    dic2['out' + str(i)], dic1['out' + str(i)] = affine_forward(dic2['relu_' + str(i-1)], self.params['W' + str(i)], self.params['b' + str(i)])
                dic2['relu_' + str(i)], dic1['relu_' + str(i)] = relu_forward(dic2['out' + str(i)])
                if self.use_dropout:
                    dic2['dropout_' + str(i)], dic1['dropout_' + str(i)] = dropout_forward(dic2['relu_' + str(i)], self.dropout_param)        
        if self.use_dropout:
            scores, dic1['out' + str(self.num_layers)] = affine_forward(dic2['dropout_' + str(self.num_layers - 1)], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        else:
            scores, dic1['out' + str(self.num_layers)] = affine_forward(dic2['relu_' + str(self.num_layers - 1)], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss1, output1 = softmax_loss(scores, y)
        loss2 = 0
        for i in range(self.num_layers):
            loss2 += np.sum(np.square(self.params['W' + str(i+1)]))
        loss2 = 0.5 * self.reg * loss2
        
        k1, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(output1, dic1['out' + str(self.num_layers)])
        if self.use_dropout:
            k1 = dropout_backward(k1, dic1['dropout_' + str(self.num_layers - 1)])
        k1 = relu_backward(k1, dic1['relu_' + str(self.num_layers - 1)])
        if self.num_layers >= 3:
            for i in range(self.num_layers - 1, 1, -1):
                k1, grads['W2'], grads['b2'] = affine_backward(k1, dic1['out' + str(i)])
                if self.use_dropout:
                    k1 = dropout_backward(k1, dic1['dropout_' + str(i - 1)])
                k1 = relu_backward(k1, dic1['relu_' + str(i - 1)])
            unused, grads['W1'], grads['b1'] = affine_backward(k1, dic1['out1'])
        for i in range(self.num_layers):
            grads['W' + str(i+1)] += self.reg * self.params['W' + str(str(i+1))]

        loss = loss1 + loss2


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
