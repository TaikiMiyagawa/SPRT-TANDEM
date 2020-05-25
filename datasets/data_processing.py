import os
import tensorflow as tf

def read_tfrecords_nosaic_mnist(record_file_train, record_file_test, batch_size, 
    shuffle_buffer_size=10000):
    """Reads TFRecord file and make parsed dataset tensors. 
       Returns train, validation and test dataset tensors with 5:1:1 split.
    Args:
        record_file_train: A string. Path to training Nosaic MNIST tfrecord file.
        record_file_test: A string. Path to test Nosaic MNIST tfrecord file.
        batch_size: An int.
        shuffle_buffer_size: An int. 
            Larger size leads to larger CPU memory consumption.
    Return:
        parsed_image_dataset_train: A dataset tensor.
        parsed_image_dataset_valid: A dataset tensor.
        parsed_image_dataset_test: A dataset tensor.
    Example:
        # Training loop
        for i, feats in enumerate(parsed_image_dataset):
            video_batch = tf.io.decode_raw(feats['video'], tf.uint8)
            video_batch = tf.cast(video_batch, tf.float32)
            video_batch = tf.reshape(video_batch, (-1, 20, 28, 28, 1)) 
                # (B, T, H, W, C)
            label_batch = tf.cast(feats["label"], tf.int32) # (B,)        
        # That is,
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_nosaic_mnist(feats)
    Example 2:
        # Training loop
        for i, feats in enumerate(parsed_image_dataset):
            video_batch = tf.io.decode_raw(feats['video'], tf.float32)
            video_batch = tf.cast(video_batch, tf.float32)
            video_batch = tf.reshape(video_batch, (-1, 128)
            label_batch = tf.cast(feats["label"], tf.int32) # (B, )        
        # That is,
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_feat(feats)
    Remark:
        drop_remainder is True to simplify 
            the training code (train_fe_nmnist.py and train_ti_nmnist.py).
    """
    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, {
                    'video': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([],tf.int64)
                    })

    raw_image_dataset = tf.data.TFRecordDataset(record_file_train)
    raw_image_dataset_test = tf.data.TFRecordDataset(record_file_test)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    parsed_image_dataset_test = raw_image_dataset_test.map(
        _parse_image_function)

    parsed_image_dataset_train = parsed_image_dataset.take(50000)
    parsed_image_dataset_valid = parsed_image_dataset.skip(50000)

    parsed_image_dataset_train = parsed_image_dataset_train.shuffle(
        shuffle_buffer_size)

    parsed_image_dataset_train = parsed_image_dataset_train.batch(
        batch_size, drop_remainder=True) 
    parsed_image_dataset_valid = parsed_image_dataset_valid.batch(
        batch_size, drop_remainder=True) 
    parsed_image_dataset_test = parsed_image_dataset_test.batch(
        batch_size, drop_remainder=True) 

    return parsed_image_dataset_train, parsed_image_dataset_valid,\
        parsed_image_dataset_test


def decode_nosaic_mnist(features):
    """Decode TFRecords.
    Returns:
        video_batch: A Tensor with shape 
            (batch, duration, height, width, channel) 
            = (-1, 20, 28 ,28 ,1) 
            that represents a batch of videos. float32.
        label_batch: A Tensor with shape (batch,) 
            that represents a batch of labels. int32.
    Examle:
        parsed_image_dataset, _, _ = read_tfrecords_nosaic_mnist(
            path, batch size)
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_nosaic_mnist(feats)
    """
    video_batch = tf.io.decode_raw(features['video'], tf.uint8)
    video_batch = tf.cast(video_batch, tf.float32)
    video_batch = tf.reshape(video_batch, (-1, 20, 28, 28, 1)) # (B, T, H, W, C)
    label_batch = tf.cast(features["label"], tf.int32) # (B, )        

    return video_batch, label_batch


def decode_feat(features, duration, feat_dim, dtype_feat=tf.float32,
    dtype_label=tf.int32):
    """Decode TFRecords.
    Returns:
        video_batch: A Tensor with shape (batch, duration, feat dim)
            that represents a batch of frames of features.
        label_batch: A Tensor with shape (batch,) 
            that represents a batch of labels. int32.
    Usage:
        parsed_image_dataset, _, _ = read_tfrecords_nosaic_mnist(path, feat dim)
        for i, feats in enumerate(parsed_image_dataset):
            video_batch, label_batch = decode_nosaic_mnist(feats)
    """
    video_batch = tf.io.decode_raw(features['video'], dtype_feat)
    video_batch = tf.reshape(video_batch, (-1, duration, feat_dim)) # (B, T, D)
    label_batch = tf.cast(features["label"], dtype_label) # (B, )        

    return video_batch, label_batch


def binarize_labels_nosaic_mnist(labels):
    """Change labels like even (class 0) vs odd (class 1) numbers
    """
    labels = labels % 2
    return labels


def normalize_images_nosaic_mnist(images):
    images /= 127.5
    images -= 1
    return images


def reshape_for_featext(x, y, feat_dims):
    """(batch, duration) to (batch * duration,)"""
    x_shape = x.shape
    batch_size = x_shape[0]
    duration = x_shape[1]

    # To disentangle, tf.reshape(x, (batch, duration, feat_dims[0]...))
    x = tf.reshape(
        x, (-1, feat_dims[0], feat_dims[1], feat_dims[2]))

    y = tf.tile(y, [duration,])
    y = tf.reshape(y, (duration, batch_size))
    y = tf.transpose(y, [1,0])
    y = tf.reshape(y, (-1,))

    return x, y


def sequential_slice(x, y, order_sprt):
    """Slice, copy, and concat a batch to make a time-sliced, augumented batch.
    Effective batch size will be batch * (duration - order_sprt)).
    e.g., nosaic MNIST and 2nd-order SPRT: 
        effective batch size is (20-2)=18 times larger 
        than the original batch size.
    Args:
        x: A Tensor with shape 
            (batch, duration, feature dimension).
        y: A Tensor with shape (batch).
        order_sprt: An int. The order of SPRT.
    Returns:
        x_slice: A Tensor with shape 
            (batch*(duration-order_sprt), order_sprt+1, feat dim).
        y_slice: A Tensor with shape 
            (batch*(duration-order_sprt),).
    Remark:
        - y_slice may be a confusing name, because we copy and concatenate
          original y to obtain y_slice.
    """
    duration = x.shape[1]
    if duration < order_sprt + 1:
        raise ValueError(
        "order_sprt must be <= duration - 1."+\
        " Now order_sprt={}, duration={} .".format(
            order_sprt, duration))

    for i in range(duration - order_sprt):
        if i == 0:
            x_slice = x[:, i:i+order_sprt+1, :]
            y_slice = y
        else:
            x_slice = tf.concat([x_slice, x[:, i:i+order_sprt+1, :]],0)
            y_slice = tf.concat([y_slice, y], 0)

    return x_slice, y_slice


def sequential_concat(x_slice, y_slice, duration=20):
    """Opposite operation of sequential_slice. 
    x_slice's shape will change 
    from (batch * (duration - order_sprt), order_sprt + 1, feat dim )
    to  (batch, (duration - order_sprt), order_sprt + 1, feat dim).
    y changes accordingly.
    Args:
        x_slice: A Tensor with shape 
            (batch * (duration - order_sprt), order_sprt + 1, feat dim). 
            This is the output of 
            models.backbones_lstm.LSTMModel.__call__(inputs, training). 
        y_slice: A Tensor with shape (batch*(duration - order_sprt),).
        duration: An int. 20 for nosaic MNIST.
    Returns:
        x_cocnat: A Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, feat dim).
        y_concat: A Tensor with shape (batch).
    Remark:
        - y_concat may be a confusing name, because we slice 
          the input argument y_slice to get y_concat.
    """
    x_shape = x_slice.shape
    order_sprt = int(x_shape[1] - 1)
    batch = int(x_shape[0] / (duration - order_sprt))
    feat_dim = x_shape[-1]

    # Cancat time-sliced, augumented batch
    x_concat = tf.reshape(
        x_slice, 
        (duration - order_sprt, batch, order_sprt + 1, feat_dim))
    x_concat = tf.transpose(x_concat, [1, 0, 2, 3]) 
    # (batch, duration - order_sprt, order_sprt + 1, feat_dim)
    y_concat = y_slice[:batch]

    return x_concat, y_concat