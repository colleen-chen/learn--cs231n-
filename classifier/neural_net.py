from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
#from past.builtins import xrange

class forward_backward_propogation(object):
    
    def __init__(self, W1, b1, W2, b2, X=0):
        """
        Only for two layer neural net
        """
        # get dimensions of nodes at each layer
        self.X = X

        self.X = 0.0                                                  # dim (N0, D)
        self.W1 = W1                                                  # dim (D, N1)   
        self.b1 = b1                                                  # dim (1, N1)
        self.z1 = 0.0                                                 # dim (N0, N1)
        self.a1 = 0.0                                                 # dim (N0, N1)
        self.W2 = W2                                                  # dim (N1, N2)
        self.b2 = b2                                                  # dim (1, N2)
        self.z2 = 0.0                                                 # dim (N0, N2)
        self.a2 = 0.0                                                 # dim (N0, N2)
        self.dz1 = 0.0                                                # dim (N0, N1)
        self.dW1 = 0.0                                                # dim (D, N1)
        self.db1 = 0.0                                                # dim (1, N1)  
        self.da1 = 0.0                                                # dim (N0, N2)
        self.dz2 = 0.0                                                # dim (N0, N2)
        self.dW2 = 0.0                                                # dim (N1, N2)
        self.db2 = 0.0                                                # dim (1, N2)
        self.da2 = 1.0

    def start(self, X):
        print('at start : {}'.format(X))
        self.X = X             
        
        N0 = self.X.shape[0]
        D = self.X.shape[1]
        N1 = self.W1.shape[1]
        N2 = self.W2.shape[1]

        print('before setting z1')
        self.z1 = np.zeros((N0, N1))                                  # dim (N0, N1)</font>
        self.a1 = np.zeros((N0, N1))                                  # dim (N0, N1)</font>
        self.z2 = np.zeros((N0, N2))                                  # dim (N0, N2)</font>
        self.a2 = np.zeros((N0, N2))                                  # dim (N0, N2)</font>
        self.dz1 = np.zeros((N0, N1))                                 # dim (N0, N1)</font>
        self.dW1 = np.zeros((D, N1))                                  # dim (D, N1)</font>
        self.db1 = np.zeros((1, N1))                                  # dim (1, N1)  </font>
        self.da1 = np.zeros((N0, N2))                                 # dim (N0, N2)</font>
        self.dz2 = np.zeros((N0, N2))                                 # dim (N0, N2)</font>
        self.dW2 = np.zeros((N1, N2))                                 # dim (N1, N2)</font>
        self.db2 = np.zeros((1, N2))                                  # dim (1, N2)</font>
        self.da2 = 1.0

    def forward_scores(self, X, W, b):
        """
        X:  Input features; has shape (N, D)
        W: First layer weights; has shape (D, H)
        b: First layer biases; has shape (1, H)
        z: output matrix; has shape (N, H)
        """
        z = X.dot(W) + b
        return z

    def backward_scores(self, dz):
        dW = self.X.T.dot(dz)
        db = dz
        return (dW, db)

    def forward_activation_softmax(self, z, y=None):
        """
        z:  scores matrix; has shape (N, H)
        y:  actual labels; has shape (1, H)
        """
        f = np.exp(z)
        # summing over all hidden nodes H in the layer
        sum_f = np.sum(f, axis=1).reshape(-1,1)
        # activation function - softmax
        a = np.zeros((z.shape[0], z.shape[1]))
        return np.exp(z) / sum_f

    def backward_activation_softmax(self, da, a):
        grad_dadz = a - a * a
        return grad_dadz.dot(da)

    def forward_activation_ReLU(self, z):
        return np.maximum(0, z)
    
    def backward_activation_ReLU(self, da, a):
        result = np.zeros((a.shape[0], a.shape[1]))
        for i in range(len(a)):
            for j in range(len(a[i])):
                if a[i,j] > 0.0:
                    result[i,j] = 1.0
                else:
                    result[i,j] = 0.0

        return result.dot(da)

def func_sigmold(x):
    return 1.0 / (1.0 + np.exp(-x))

def func_softmax(x, i):
    """
    one dimensional x
    """
    f = np.exp(x)
    sum_f = np.sum(f, axis=1).reshape(-1,1)
    return np.exp(x[i]) / sum_f

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (1, H)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (1, C)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.twolayerNN = forward_backward_propogation(self.params['W1'], self.params['b1'],
                                                       self.params['W2'], self.params['b2'])

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        self.twolayerNN.start(X)
        #############################################################################
        # DONE: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # the score at 1st layer, with dimension (N, H)
        self.twolayerNN.z1 = self.twolayerNN.forward_scores(X, W1, b1)
        # activation a_1 at 1st layer after applying activation function softmax
        self.twolayerNN.a1 = self.twolayerNN.forward_activation_softmax(self.twolayerNN.z1)
        # the scores at 2nd layer, with dimension (H, C)
        self.twolayerNN.z2 = self.twolayerNN.forward_scores(self.twolayerNN.a1, W2, b2)
        # function a_2 at 2nd layer after applying activation function
        self.twolayerNN.a2 = self.twolayerNN.forward_activation_ReLU(self.twolayerNN.z2)
        scores = self.twolayerNN.a2
        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # DONE: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # first hidden layer, using sigmold activation function
        lambda1 = 0.0
        lambda2 = 0.0
        loss = np.sum(((-1.0)*y.dot(np.log(scores)) - (1 - y).dot(np.log(abs( - scores)))), axis=0) + \
                    lambda1*np.sum(np.sum(W1.T.dot(W1), axis=1), axis=0) + \
                    lambda2*np.sum(np.sum(W2.T.dot(W2), axis=1), axis=0)

        #pass
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # DONE: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # backward propogation
        # gradient of a_2
 #       self.twolayerNN.da2 = 
        # gradient of z2
        self.twolayerNN.dz2 = self.twolayerNN.backward_activation_ReLU(self.twolayerNN.da2, self.twolayerNN.a2)
        # gradient dW2, db2
        self.twolayerNN.dW2, self.twolayerNN.db2 = self.twolayerNN.backward_scores(self.twolayerNN.dz2)

        # gradient of a1
        self.twolayerNN.da1 = W2
        # gradient of z1
        self.twolayerNN.dz1 = self.twolayerNN.backward_activation_softmax(self.twolayerNN.da1, self.twolayerNN.a1) 
        # gradient of dW1, db1
        self.twolayerNN.dW1, self.twolayerNN.db1 = self.twolayerNN.backward_scores(self.twolayerNN.dz1)
        
        grads['W1'] = self.twolayerNN.dW1
        grads['b1'] = self.twolayerNN.db1
        grads['W2'] = self.twolayerNN.dW2
        grads['b2'] = self.twolayerNN.db2
        #pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO Create a random minibatch of training data and labels, storing  # 
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
