from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
class ThreeLayerNet(object):
  """
  A three-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses ReLU nonlinearities after the first and the second fully
  connected layers.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the third fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, H)
    b2: Second layer biases; has shape (H,)
    W3: Third layer weights; has shape (H, C)
    b3: Third layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, hidden_size)
    self.params['b2'] = np.zeros(hidden_size)
    self.params['W3'] = std * np.random.randn(hidden_size, output_size)
    self.params['b3'] = np.zeros(output_size)


  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a three layer fully connected neural
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
    W3, b3 = self.params['W3'], self.params['b3']
    N, D = X.shape
    C = b3.shape[0]

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    z1 = np.dot(X, W1) + b1   #全连接层1 (N, H)
    h1 = np.maximum(0, z1)        # ReLU
    z2 = np.dot(h1, W2) + b2  #全连接层2 (N, H)
    h2 = np.maximum(0, z2)        # ReLU
    z3 = np.dot(h2, W3) + b3  #全连接层3 (N, C)
    scores = z3
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1, W2, and W3. Store the    #
    # result in the variable loss, which should be a scalar. Use the Softmax    #
    # classifier loss.                                                          #
    #############################################################################
    # shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1) #keeepdims亦可
    # exp_scores = np.exp(shift_scores)
    # softmax_out = exp_scores / np.num(exp_scores, axis=1, keepdims=True)  # (N, C)
    # loss = np.sum(-np.log(softmax_out[range(N), y])) / N + 0.5 * reg * (np.sum(W1 * W1) + np.num(W2 * W2) + np.num(W3 * W3))
    # print("loss %f" % loss)
    # 化简版公式
    f = scores - np.max(scores, axis=1, keepdims=True)
    loss = -f[range(N), y].sum() + np.log(np.exp(f).sum(axis=1)).sum()
    loss = loss / N + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W2 * W2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dscores = np.exp(f) / np.exp(f).sum(axis=1, keepdims=True) #(N, C)
    dscores[range(N), y] -= 1             #(N, C) 只操作了输出正确的位置，其余没有动
    dscores /= N      #FIXME: minbatch内平均? 可是后面也dh,dW也没求和啊？
    # Third  Layer
    dh2 = np.dot(dscores, W3.T)       # (N, H)
    dW3 = np.dot(h2.T, dscores)       # (H, C)
    db3 = np.sum(dscores, axis=0)     # (C, )
    #ReLU
    dh2[z2<=0] = 0
    dz2 = dh2         # (N, H)
    # Second Layer
    dh1 = np.dot(dz2, W2.T)     # (N, H)
    dW2 = np.dot(h1.T, dz2)     # (H, H) 这两个H分别是HiddenLayer1,2的神经元数目
    db2 = np.sum(dz2, axis=0)   # (H, )
    # ReLU
    dh1[z1 <= 0] = 0
    dz1 = dh1  # (N, H)
    # First Layer
    dW1 = np.dot(X.T, dz1)      # (D, H)
    db1 = np.sum(dz1, axis=0)   # (H, )
    # Reg
    grads['W3'] = dW3 + reg * W3
    grads['b3'] = db3
    grads['W2'] = dW2 + reg * W2
    grads['b2'] = db2
    grads['W1'] = dW1 + reg * W1
    grads['b1'] = db1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
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
    # epoch: 所有training data 遍历一次的更新次数
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    print("iterations_per_epoch = : %d" %(iterations_per_epoch))

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    count = 0
    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_mask = np.random.choice(num_train, batch_size, replace=True) # 被抽取样本可重复出现
      X_batch = X[batch_mask]
      y_batch = y[batch_mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      self.params['W3'] -= learning_rate * grads['W3']
      self.params['b3'] -= learning_rate * grads['b3']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################
      # count += 1
      # if count % 50 == 0:
      #   print("count is :{}".format(count))
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
        count += 1
        print("the %rd epoch completed!" %(count))

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this three-layer network to predict labels for
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
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    h1 = np.maximum(0, np.dot(X, W1) + b1)
    h2 = np.maximum(0, np.dot(h1, W2) + b2)
    scores = np.dot(h2, W3) + b3
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


