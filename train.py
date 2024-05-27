#import package
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import time

from utils.LayerObjects import *
from utils.utils_func import *

# the path of the dataset
test_image_path = r'./MNIST/t10k-images.idx3-ubyte'
test_label_path = r'./MNIST/t10k-labels.idx1-ubyte'
train_image_path = r'./MNIST/train-images.idx3-ubyte'
train_label_path = r'./MNIST/train-labels.idx1-ubyte'
trainset = (train_image_path, train_label_path)
testset = (test_image_path, test_label_path)

# read the dataset with readDataset()
(train_image, train_label) = readDataset(trainset)
(test_image, test_label) = readDataset(testset)
n_m, n_m_test = len(train_label), len(test_label)
print("The shape of training image:", train_image.shape)
print("The shape of testing image: ", test_image.shape)
print("Length of the training set: ", n_m)
print("Length of the training set: ", n_m_test)
print("Shape of a single image: ", train_image[0].shape)

train_image_normalized_pad = normalize(zero_pad(train_image[:,:,:,np.newaxis], 2),'lenet5')
test_image_normalized_pad  = normalize(zero_pad(test_image[:,:,:,np.newaxis],  2),'lenet5')
print("The shape of training image with padding:", train_image_normalized_pad.shape)
print("The shape of testing image with padding: ", test_image_normalized_pad.shape)


# The fixed weight (7x12 preset ASCII bitmaps) used in the RBF layer.
bitmap = rbf_init_weight()
fig, axarr = plt.subplots(2,5,figsize=(20,8))
for i in range(10):
    x,y = int(i/5), i%5
    axarr[x,y].set_title(str(i))
    axarr[x,y].imshow(bitmap[i,:].reshape(12,7), cmap=mpl.cm.Greys)


ConvNet = LeNet5()

# Number of epoches & learning rate in the original paper
epoch_orig, lr_global_orig = 20, np.array([5e-4]*2 + [2e-4]*3 + [1e-4]*3 + [5e-5]*4 + [1e-5]*8)

# Number of epoches & learning rate I used
epoches, lr_global_list = epoch_orig, lr_global_orig*100

momentum = 0.9
weight_decay = 0
batch_size = 256

# Training loops
st = time.time()
cost_last, count = np.Inf, 0
err_rate_list = []
for epoch in range(0, epoches):
    print("---------- epoch", epoch + 1, "begin ----------")

    # Stochastic Diagonal Levenberg-Marquaedt method for determining the learning rate
    (batch_image, batch_label) = random_mini_batches(train_image_normalized_pad, train_label, mini_batch_size=500,
                                                     one_batch=True)
    ConvNet.Forward_Propagation(batch_image, batch_label, 'train')
    lr_global = lr_global_list[epoch]
    ConvNet.SDLM(0.02, lr_global)

    # print info
    print("global learning rate:", lr_global)
    print("learning rates in trainable layers:", np.array([ConvNet.C1.lr, ConvNet.C3.lr, ConvNet.C5.lr, ConvNet.F6.lr]))
    print("batch size:", batch_size)
    print("Momentum:", momentum, ", weight decay:", weight_decay)

    # loop over each batch
    ste = time.time()
    cost = 0
    mini_batches = random_mini_batches(train_image_normalized_pad, train_label, batch_size)
    for i in range(len(mini_batches)):
        batch_image, batch_label = mini_batches[i]

        loss = ConvNet.Forward_Propagation(batch_image, batch_label, 'train')
        cost += loss

        ConvNet.Back_Propagation(momentum, weight_decay)

        # print progress
        if i % (int(len(mini_batches) / 100)) == 0:
            sys.stdout.write("\033[F")  # CURSOR_UP_ONE
            sys.stdout.write("\033[K")  # ERASE_LINE
            print("progress:", int(100 * (i + 1) / len(mini_batches)), "%, ", "cost =", cost, end='\r')
    sys.stdout.write("\033[F")  # CURSOR_UP_ONE
    sys.stdout.write("\033[K")  # ERASE_LINE

    print("Done, cost of epoch", epoch + 1, ":", cost, "                                             ")

    error01_train, _ = ConvNet.Forward_Propagation(train_image_normalized_pad, train_label, 'test')
    error01_test, _ = ConvNet.Forward_Propagation(test_image_normalized_pad, test_label, 'test')
    err_rate_list.append([error01_train / 60000, error01_test / 10000])
    print("0/1 error of training set:", error01_train, "/", len(train_label))
    print("0/1 error of testing set: ", error01_test, "/", len(test_label))
    print("Time used: ", time.time() - ste, "sec")
    print("---------- epoch", epoch + 1, "end ------------")
    with open('model_data_' + str(epoch) + '.pkl', 'wb') as output:
        pickle.dump(ConvNet, output, pickle.HIGHEST_PROTOCOL)

err_rate_list = np.array(err_rate_list).T
print("Total time used: ", time.time() - st, "sec")

# This shows the error rate of training and testing data after each epoch
x = np.arange(epoches)
plt.xlabel('epoches')
plt.ylabel('error rate')
plt.plot(x, err_rate_list[0])
plt.plot(x, err_rate_list[1])
plt.legend(['training data', 'testing data'], loc='upper right')
plt.show()

# read model
with open('model_data_13.pkl', 'rb') as input_:
    ConvNet = pickle.load(input_)

test_image_normalized_pad = normalize(zero_pad(test_image[:,:,:,np.newaxis], 2), 'lenet5')
error01, class_pred = ConvNet.Forward_Propagation(test_image_normalized_pad, test_label, 'test')
#print(class_pred)
print("error rate:", error01/len(class_pred))