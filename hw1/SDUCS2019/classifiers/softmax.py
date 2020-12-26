import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y = np.asarray(y, dtype=np.int)
  num_train, dim = X.shape
  num_class = W.shape[1]
  score = X.dot(W)
  # 求每个样本的最大输出得分
  score_max = np.max(score, axis=1).reshape(num_train, 1)
  # 计算对数概率，prob.shape = N*D，每行对应一个样本
  # 分子分母同乘常数C，防止指数溢出，取log(C) = -score_max
  z = np.sum(np.exp(score - score_max), axis=1, keepdims=True)  # N行1列，keepdims使消失的轴保留，保持其维度为1
  z1 = np.sum(np.exp(score - score_max), axis=1)                # 相加的那个维度会消失，变成一维数组了
  e_j = np.exp(score - score_max)
  prob = e_j / z  # N行1列?
  print(type(prob))
  for i in range(num_train):
    loss += -np.log(prob[i, y[i]])
    for j in range(num_class):
      if j == y[i]:
        dW[:, j] += -(1 - prob[i, j]) * X[i]
      else:
        dW[:, j] += -(0 - prob[i, j]) * X[i]
  # minibatch内取平均
  loss = loss / num_train +0.5 * reg * np.sum(W*W)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO:
  # Compute the softmax loss and its gradient using no explicit loops.        #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # num_train = X.shape[0]
  # num_class = W.shape[1]
  # y = np.asarray(y, dtype=np.int)
  # for i in range(num_train):
  #   s = X[i].dot(W)
  #   score = s - np.max(s)     # (1, C)
  #   score_E = np.exp(score)
  #   Z = np.sum(score_E)            # scalar
  #   score_target = score_E[y[i]]    # scalar
  #   loss += -np.log(score_target / Z)   # scalar
  #   for j in range(num_class):
  #     if j == y[i]:
  #       dW[:, j] += -X[i] * (1 - score_E[j] / Z)
  #     else:
  #       dW[:, j] += -X[i] * (0 - score_E[j] / Z)
  #
  # loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  # dW = dW / num_train + reg * W

  N = X.shape[0]
  z = np.dot(X, W)          # (N, C) 同一行是一个样本
  z -= z.max(axis=1, keepdims=True)   # 广播。zmax:(N, 1)。即每一列都减去了最大值
  sz = np.exp(z).sum(axis=1, keepdims=True)  #(N, 1) 每一行求和
  #assert s.shape == (N, 1)
  loss = np.log(sz).sum() - z[range(N), y].sum()   #sum所有样本,化简后公式
  score = np.exp(z) / sz     # out of softmax layer (N, C)
  # grad
  dscore = score            # j!=y[i]
  dscore[range(N), y] -= 1  # j==y[i]   (N, C)
  dW = np.dot(X.T, dscore)  # backpropagation X(N, D) dW(D, C)

  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

