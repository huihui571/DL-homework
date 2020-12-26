import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  两层的全连接网络。使用sotfmax损失函数和L2正则，非线性函数采用Relu函数。
  网络结构：input - fully connected layer - ReLU - fully connected layer - softmax
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    初始化模型。
    初始化权重矩阵W和偏置b。这里b置为零，但是Alexnet论文中说采用Relu函数激活时b置为1可以更快的收敛。
    参数都保存在self.params字典中。
    键为：
    W1 (D, H)
    b1 (H,)
    W2 (H, C)
    b2 (C,)
    D,H,C分别表示输入数据的维度，隐藏层大小，输出类别的个数
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)


  def loss(self, X, y=None, reg=0.0):
    """
    如果是在训练过程,计算损失和梯度，如果是在测试过程，返回最后一层的输入,即每个类的得分。

    Inputs:
    - X (N, D).  X[i] 为一个训练样本。
    - y: 标签。如果为None则表示是在进行测试过程，否则是在进行训练过程。
    - reg: Regularization strength.

    Returns:
    如果y=None，返回shape为(N, C)的矩阵，scores[i, c]表示输入i在c类上的得分。

    如果y!=None, 返回一个tuple:
    - loss: 包括数据损失和正则损失两部分。
    - grads: 各个参数的梯度。
    """
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    C=b2.shape[0]
    
    #forward pass
    h1=np.maximum(0,np.dot(X,W1)+b1)
    h2=np.dot(h1,W2)+b2
    scores=h2
    
    if y is None:
      return scores

    # 计算loss
    shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)
    exp_scores=np.exp(shift_scores)
    softmax_out=exp_scores/np.sum(exp_scores,axis=1).reshape(-1,1)
    loss=np.sum(-np.log(softmax_out[range(N),y]))/N+reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    print(np.sum(-np.log(softmax_out[range(N),y]))/N,reg * (np.sum(W1 * W1) + np.sum(W2 * W2)))

    # Backward pass: 计算梯度，梯度的计算就是链式求导的过程
    grads = {}

    dscores = softmax_out.copy()
    dscores[range(N),y]-=1
    dscores /= N
    
    grads['W2']=np.dot(h1.T,dscores)+2*reg*W2
    grads['b2']=np.sum(dscores,axis=0)
    
    dh=np.dot(dscores,W2.T)
    d_max=(h1>0)*dh
    
    grads['W1'] = X.T.dot(d_max) + 2*reg * W1
    grads['b1'] = np.sum(d_max, axis = 0)

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    自动化训练过程。采用SGD优化。

    Inputs:
    - X (N, D)：训练输入。
    - y (N,) ：标签。 y[i] = c 表示X[i]的类别下标是c。
    - X_val (N_val, D)：验证集输入。
    - y_val (N_val,)： 验证集标签。
    - learning_rate: 
    - learning_rate_decay: 学习率的损失因子。
    - reg: regularization strength。
    - num_iters: 迭代次数。
    - batch_size: 每次迭代的数据批大小。.
    - verbose: 是否显示训练进度。
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      #随机选择一批数据
      idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[idx]
      y_batch = y[idx]
      # 计算损失和梯度
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      #更新参数
      self.params['W2'] += - learning_rate * grads['W2']
      self.params['b2'] += - learning_rate * grads['b2']
      self.params['W1'] += - learning_rate * grads['W1']
      self.params['b1'] += - learning_rate * grads['b1']
      #可视化进度
      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # 每个epoch保存一次数据记录
      if it % iterations_per_epoch == 0:
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        #学习率衰减
        learning_rate *= learning_rate_decay
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    使用训练好的参数预测输入的标签。

    Inputs:
    - X (N, D)： 需要预测的输入。

    Returns:
    - y_pred (N,)：每个输入的预测分类下标。
    """
    
    h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
    scores = h.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)

    return y_pred