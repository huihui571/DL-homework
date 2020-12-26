# coding: utf-8

# 实现一个简单的神经网络并在CIFAR10上测试性能

import numpy as np
import matplotlib.pyplot as plt
from neural_net import TwoLayerNet
from data_utils import load_CIFAR10
from vis_utils import visualize_grid

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # 采样
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 归一化操作：减去均值，使得数据以0为中心
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


#第一次训练
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

#效果不太理想，debug

# 先画一下loss和正确率的曲线看一看
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.show()



#可视化一下权重
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)


#通过上面的曲线我们可以看到基本上loss还在线性下降，表示我们的loss下降的还不够。
#一方面，我们可以加大学习率使loss更加快速的下降，另一方面，也可以增加迭代的次数，让loss继续下降。
#还有，在训练集和验证集上的正确率没有明显差距，表明网络的容量可能不够，可以尝试增加网络的复杂度使之拥有更强的表达能力。



#下面是我调出来的参数，实际上选了很久 ,在测试集上的正确率在55%左右
hidden_size = 150#[50,70,100,130]
learning_rates = 1e-3#np.array([0.5,1,1.5])*1e-3
regularization_strengths = 0.2#[0.1,0.2,0.3]
best_net = None
results = {}
best_val_acc = 0


for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            
            net = TwoLayerNet(input_size, hs, num_classes)
            # Train the network
            stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=3000, batch_size=200,
            learning_rate=lr, learning_rate_decay=0.95,
            reg= reg, verbose=False)
            val_acc = (net.predict(X_val) == y_val).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = net         
            results[(hs,lr,reg)] = val_acc
            
            plt.subplot(2, 1, 1)
            plt.plot(stats['loss_history'])
            plt.title('Loss history')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')

            plt.subplot(2, 1, 2)
            plt.plot(stats['train_acc_history'], label='train')
            plt.plot(stats['val_acc_history'], label='val')
            plt.title('Classification accuracy history')
            plt.xlabel('Epoch')
            plt.ylabel('Clasification accuracy')
            plt.show()


for hs,lr, reg in sorted(results):
    val_acc = results[(hs, lr, reg)]
    print ('hs %d lr %e reg %e val accuracy: %f' % (hs, lr, reg,  val_acc))
    
print ('best validation accuracy achieved during cross-validation: %f' % best_val_acc)


show_net_weights(best_net)
test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)