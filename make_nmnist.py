# tf.__version__ = 2.0.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

############## USER DEFINED #####################
data_dir = "/data-directory/tensorflow_datasets" 
    # to be downloaded to this directory
train_or_test = "train" 
    # "train" (train dataset will be made) or "test" (test dataset will be made). 
    # Make sure to execute this script twice. 
    # One is to make the training dataset; the othrer is to make the test dataset.
record_file = "/data-directory/nosaic_mnist/nosaic_mnist_{}.tfrecords".format(train_or_test)
    # TFR name
#################################################

# Functions
def set_gpu_devices(gpu):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(physical_devices[gpu], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu], True)

def np_to_tfr_nosaic_mnist(x, y, writer):
    """Save a np.array to a tfrecord file. DO NOT FORGET writer.close().
    Args:
        x: data: np.ndarray, dtype=uint8
        y: label: int, dtype=int64
        writer: tf.io.TFRecordWriter object. Don't forget writer.close()
    """
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # Make an Example object that has one record data
    example = tf.train.Example(features=tf.train.Features(feature={
        'video': _bytes_feature(x.tostring()),
        'label': _int64_feature(y)
        }))

    # Serialize the example object and make a TFRecord file
    writer.write(example.SerializeToString())

def fix_random_seed(flag_seed, seed=None):
    if flag_seed:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print("Numpy and TensorFlow's random seeds fixed: seed=" + str(seed))
    
    else:
        print("Random seed not fixed.")
    

set_gpu_devices(0) # the first GPU (numbered 0) will be used
fix_random_seed(True, 7)

# Load MNIST as numpy array
############################
dstr, dsts = tfds.load(name="mnist", data_dir=data_dir, split=["train", "test"], batch_size=-1)
images_train = dstr["image"].numpy() # (60000, 28, 28, 1), np.uint8
images_test = dsts["image"].numpy()  # (60000,), np.int64
labels_train = dstr["label"].numpy() # (10000, 28, 28, 1), np.uint8
labels_test = dsts["label"].numpy()  # (10000,), np.int64

# train or test
if train_or_test == "train":
    images_make = images_train
    labels_make = images_train
elif train_or_test == "test":
    images_make = images_test
    labels_make = labels_test
else:
    raise ValueError(train_or_test)

# Make training "OR" test data of nosaic MNIST
############################################################
if os.path.exists(record_file):
    raise ValueError("record_file exists.")
    
with tf.io.TFRecordWriter(record_file) as writer:
#with tf.io.TFRecordWriter(record_file, options=tf.io.TFRecordOptions(compression_type=tf.compat.v1.io.TFRecordCompressionType.GZIP)) as writer: #####
    cnt = 1
    for image, label in zip(images_make, labels_make):
        # Verbose
        if cnt % 100 == 0:
            print("Iteration {}/{}".format(cnt, len(images_make)))
        cnt += 1

        # Reshape    
        image_reshape = np.reshape(image, (28, 28))

        # 1. Generate a sequence of offsets of nosaic
        ########################################
        # Remove 40 or 28 noisy pixels in each step (40 * 19 + 28 = 784(= 28*28*1))
        idx_perm = np.random.permutation(784) 
        idx_perm_split = [idx_perm[40*k: 40*k+40] for k in range(0, 20)] # (20, 28 if last cell (offset=19) else 40 )

        # 2. Make masks that are filled with 255
        ########################################
        masks = [None]*20
        bkgd = np.array([255]*784, dtype=np.uint32)
        for i, offsets in enumerate(idx_perm_split):
            if i == 0:
                masks[i] = bkgd - 0
            else:
                masks[i] = masks[i-1] - 0 
            masks[i][offsets] = masks[i][offsets] - 255

        # Reshape
        for i, mask in enumerate(masks):
            masks[i] = np.reshape(mask, (28,28))

        # 3. Synthesize the mask and the image
        ########################################
        masked_images = [None]*20
        image_org = np.uint32(image_reshape)

        for i, mask in enumerate(masks):
            masked_images[i] = np.uint8(np.clip(image_org + mask, 0, 255))

        # Reshape
        video = np.reshape(masked_images, (20, 28, 28, 1)) # from (20,28,28) to (20,28,28,1)

        # 4. Save images
        ########################################
        label = np.int64(label) 
        np_to_tfr_nosaic_mnist(x=video, y=label, writer=writer)

        
""" APPENDIX: Use this (works on jupyter notebook) to check the TFRecord generated."""    
def read_tfr(record_file='/data-directory/nosaic_mnist/nosaic_mnist_test.tfrecords'):
    # Read TFR
    def _parse_image_function(example_proto):
      return tf.io.parse_single_example(example_proto, {
                'video': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([],tf.int64)
                })

    raw_image_dataset = tf.data.TFRecordDataset(record_file) 
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset = parsed_image_dataset.shuffle(40000)
    parsed_image_dataset = parsed_image_dataset.batch(128, drop_remainder=False)

    tic = time.time()
    for i, feats in enumerate(parsed_image_dataset):
        video_batch = tf.io.decode_raw(feats['video'], tf.uint8)
        video_batch = tf.cast(video_batch, tf.float32)
        video_batch = tf.reshape(video_batch, (-1, 20, 28, 28, 1)) # (B, T, H, W, C)
        label_batch = tf.cast(feats["label"], tf.int32) # (B, )
    print(time.time() - tic)
    
    # Plot (works on jupyter notebook)
    cnt = 0
    for video, label in zip(video_batch, label_batch):
        fig = plt.figure(figsize=(28, 28), facecolor="w")
        for i, img in enumerate(video):
            img = np.reshape(img, (28, 28))
            ax = fig.add_subplot(20, 1, i+1)
            ax.imshow(img, cmap="gray")
            ax.set_axis_off()

        plt.show() # rewrite around here if necessary
        print(label.numpy())
        cnt += 1
        if cnt == 5:
            break