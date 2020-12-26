###################################

# 请根据需求自己补充头文件、函数体输入参数。
import numpy as np
import math
import torch
import torch.nn.functional as F


###################################
# 2 Vectorization
###################################

def vectorize_sumproducts(array1_1, array1_2):
    """
     Takes two 1-dimensional arrays and sums the products of all the pairs.
    :return:
    """
    return np.sum(np.array(array1_1) * np.array(array1_2))


def vectorize_Relu(array2):
    """
    Takes one 2-dimensional array and apply the relu function on all the values of the array.
    :return:
    """
    return np.where(np.array(array2) < 0, 0, np.array(array2))


def vectorize_PrimeRelu(array2):
    """
    Takes one 2-dimensional array and apply the derivative of relu function on all the values of the array.
    :return:
    """
    return np.where(np.array(array2) > 0, 1, 0)


######################################
# 3 Variable length
######################################

# Slice

def Slice_fixed_point(array3, start, length):
    """
    Takes one 3-dimensional array with the starting position and the length of the output instances.
    Your task is to slice the instances from the same starting position for the given length.
    :return:
    """
    return np.array([[i1[start - 1:start + length] for i1 in i2] for i2 in array3])


def slice_last_point(array3, last):
    """
     Takes one 3-dimensional array with the length of the output instances.
     Your task is to keeping only the l last points for each instances in the dataset.
    :return:
    """
    return np.array([[i1[-last:] for i1 in i2] for i2 in array3])


def slice_random_point(array3):
    """
     Takes one 3-dimensional  array  with  the  length  of the output instances.
     Your task is to slice the instances from a random point in each of the utterances with the given length.
     Please use function numpy.random.randint for generating the starting position.
    :return:
    """
    return [[i1[np.random.randint(0, len(i1)):] for i1 in i2] for i2 in array3]


# Padding

def pad_pattern_end(array3):
    """
    Takes one 3-dimensional array.
    Your task is to pad the instances from the end position as shown in the example below.
    That is, you need to pad the reflection of the utterance mirrored along the edge of the array.
    :return:
    """
    max_length = np.max([[len(i1) for i1 in i2] for i2 in array3])
    return np.array([[np.pad(i1, ([0, max_length - len(i1)]), 'symmetric') for i1 in i2] for i2 in array3])


def pad_constant_central(array3, c):
    """
     Takes one 3-dimensional array with the constant value of padding.
     Your task is to pad the instances with the given constant value while maintaining the array at the center of the padding.
    :return:
    """
    max_length = np.max([[len(i1) for i1 in i2] for i2 in array3])
    return np.array([[np.pad(i1, (
        [max_length - len(i1) - math.ceil((max_length - len(i1)) / 2), math.ceil((max_length - len(i1)) / 2)]),
                             'constant', constant_values=c) for i1 in i2] for i2 in array3])


#######################################
# PyTorch

#######################################

# numpy&torch

def numpy2tensor(array2):
    """
    Takes a numpy ndarray and converts it to a PyTorch tensor.
    Function torch.tensor is one of the simple ways to implement it but please do not use it this time.
    :return:
    """
    return torch.from_numpy(array2)


def tensor2numpy(t_array2):
    """
    Takes a PyTorch tensor and converts it to a numpy ndarray.
    :return:
    """
    return t_array2.numpy()


# Tensor Sum-products

def Tensor_Sumproducts(t_array1_1, t_array1_2):
    """
    you are to implement the function tensor sumproducts that takes two tensors as input.
    returns the sum of the element-wise products of the two tensors.
    :return:
    """
    return t_array1_1.mul(t_array1_2).sum()


# Tensor ReLu and ReLu prime

def Tensor_Relu(t_array2):
    """
    Takes one 2-dimensional tensor and apply the relu function on all the values of the tensor.
    :return:
    """
    return F.relu(t_array2)


def Tensor_Relu_prime(t_array2):
    """
    Takes one 2-dimensional tensor and apply the derivative of relu function on all the values of the tensor.
    :return:
    """
    return torch.ge(t_array2, 1)


if __name__ == '__main__':
    array1_1 = [1, 2, 3, 4]
    array1_2 = [1, 2, 3, 4]
    r = vectorize_sumproducts(array1_1, array1_2)
    print(r)
    array2 = [[2, -1, 3],
              [-1, 2, 3],
              [3, 2, -1]]
    r = vectorize_Relu(array2)
    print(r)
    r = vectorize_PrimeRelu(r)
    print(r)
    array3 = [
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8]
        ],
        [
            [9, 8, 3, 4, 5],
            [6, 7]
        ]
    ]
    r = Slice_fixed_point(array3, 1, 1)
    print(r)
    r = slice_last_point(array3, 2)
    print(r)
    r = slice_random_point(array3)
    print(r)
    r = pad_pattern_end(array3)
    print(r)
    r = pad_constant_central(array3, -1)
    print(r)
    r = numpy2tensor(np.array(array2))
    print(r)
    r = tensor2numpy(r)
    print(r)
    r = Tensor_Sumproducts(numpy2tensor(np.array(array1_1)), numpy2tensor(np.array(array1_2)))
    print(r)
    r = Tensor_Relu(numpy2tensor(np.array(array2)))
    print(r)
    r = Tensor_Relu_prime(r)
    print(r)
