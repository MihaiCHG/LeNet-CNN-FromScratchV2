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
testset = (test_image_path, test_label_path)

# read the dataset with readDataset()
(test_image, test_label) = readDataset(testset)
n_m_test = len(test_label)
print("The shape of testing image: ", test_image.shape)
print("Length of the training set: ", n_m_test)

test_image_normalized_pad  = normalize(zero_pad(test_image[:,:,:,np.newaxis],  2),'lenet5')
print("The shape of testing image with padding: ", test_image_normalized_pad.shape)


# The fixed weight (7x12 preset ASCII bitmaps) used in the RBF layer.
bitmap = rbf_init_weight()
# fig, axarr = plt.subplots(2,5,figsize=(20,8))
# for i in range(10):
#     x,y = int(i/5), i%5
#     axarr[x,y].set_title(str(i))
#     axarr[x,y].imshow(bitmap[i,:].reshape(12,7), cmap=mpl.cm.Greys)


ConvNet = LeNet5()

with open('model_data_19.pkl', 'rb') as input_:
    ConvNet = pickle.load(input_)

error01, class_pred = ConvNet.Forward_Propagation(test_image_normalized_pad, test_label, 'test')
print(class_pred)
print("error rate:", error01/len(class_pred))