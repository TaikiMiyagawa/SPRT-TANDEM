import numpy as np
import tensorflow as tf
from datasets.data_processing import sequential_slice, sequential_concat

def multiplet_loss_func(logits_slice, labels_slice):
    """Multiplet loss for density estimation of time-series data.
    Args:
        model: A model.backbones_lstm.LSTMModel object. 
        logits_slice: A logit Tensor with shape ((effective) batch size, order of SPRT + 1, num classes). This is the output of LSTMModel.call(inputs, training).
        labels_slice: A label Tensor with shape ((effective) batch size,)
    Returns:
        mpl: A scalar Tensor. Sum of multiplet losses.
    """
    # Calc multiplet losses     
    mpl = 0.
    order_sprt = logits_slice.shape[1] - 1
    for iter_Nplet in range(order_sprt + 1):
        logit = logits_slice[:, iter_Nplet, :]
        xent = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels_slice, logits=logit) # batch-averaged scalar
        mpl += xent

    return mpl


def binary_llr_loss_func(logits_concat, labels_concat):
    """Log-likelihood ratio loss for early binary classification of time-series.
    Args:
        logits_concat: A logit Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, 2). This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice)
        labels_concat: A binary label Tensor with shape (batch size,) with label = 0 or 1. This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Return:
        llr_loss: A scalar Tensor that represents the log-likelihoood ratio loss.
    Remark:
        - Binary classification (num classes = 2) is assumed.
    """
    assert tf.reduce_max(labels_concat) <= 1, "Only nb_cls=2 is allowed to use binary_llr_loss_func."

    # Start calc of LLR loss. See the N-th-order SPRT formula.
    logits_concat_shape = logits_concat.shape
    order_sprt = int(logits_concat_shape[2] - 1)
    duration = int(logits_concat_shape[1] + order_sprt)

    llr_loss = tf.constant(0.)
    for iter_frame in range(duration):
        # i.i.d. SPRT (0th-order SPRT)
        if order_sprt == 0:
            llrs_all_frames = logits_concat[:, :, order_sprt, 1] - logits_concat[:, :, order_sprt, 0] # (batch, duration-order_sprt, order_sprt+1, nb_cls=2) -> (batch, duration-order_sprt)
            llrs = tf.reduce_sum(llrs_all_frames[:, :iter_frame+1], -1) # (batch,)

            llr_losses = tf.abs(tf.cast(labels_concat, tf.float32) - tf.sigmoid(llrs)) # (batch,)
            llr_loss += tf.reduce_mean(llr_losses) # add the llr loss of the "iter_frame"-th frame

        # N-th-order SPRT
        else:
            if iter_frame < order_sprt + 1:
                llrs = logits_concat[:, 0, iter_frame, 1] - logits_concat[:, 0, iter_frame, 0] 

                llr_losses = tf.abs(tf.cast(labels_concat, tf.float32) - tf.sigmoid(llrs)) # (batch,)
                llr_loss += tf.reduce_mean(llr_losses) # the llr loss of the "iter_frame"-th frame

            else:
                llrs1 = logits_concat[:, :iter_frame - order_sprt + 1, order_sprt, 1] - logits_concat[:, :iter_frame - order_sprt + 1, order_sprt, 0] # (batch, iter_frame-order_sprt)
                llrs1 = tf.reduce_sum(llrs1, -1) # (batch,)
                llrs2 = logits_concat[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 1] - logits_concat[:, 1:iter_frame - order_sprt + 1, order_sprt-1, 0] # (batch, iter_frame-order_sprt-1)
                llrs2 = tf.reduce_sum(llrs2, -1) # (batch,)
                llrs = llrs1 - llrs2 # (batch, )

                llr_losses = tf.abs(tf.cast(labels_concat, tf.float32) - tf.sigmoid(llrs)) # (batch,)
                llr_loss += tf.reduce_mean(llr_losses) # the llr loss of the "iter_frame"-th frame

    return llr_loss


def binary_sequential_loss_func(logits_slice, labels_slice, training, duration=20):
    """Calc multiplet loss and log-likelihood ratio loss.
    Args:
        logits_slice: An logit Tensor with shape ((effective) batch size, order of SPRT, 2). This is the output of LSTMModel.__call__(inputs, training).
        labels_slice: A label Tensor with shape ((effective) batch size,) 
        training: A boolean. Training flag will be used in BatchNormalization and dropout.
    Returns:
        multiplet_loss: A list (length = order of SPRT) of scalar Tensors. Multiplet losses. 
        llr_loss: A scalar Tensor. Log-likelihood ratio loss.
        logits_concat: A logit Tensor with shape (batch, (duration - order_sprt), order_sprt + 1, num classes). This is the output of datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    Remark:
        - Binary classification (num classes = 2) is assumed.
    """
    multiplet_loss = multiplet_loss_func(logits_slice, labels_slice)
    logits_concat, labels_concat = sequential_concat(logits_slice, labels_slice, duration) 
    llr_loss = binary_llr_loss_func(logits_concat, labels_concat)

    return multiplet_loss, llr_loss, logits_concat


def get_gradient_lstm(model, x, y, training, order_sprt, duration, 
    param_multiplet_loss, param_llr_loss, param_wd, flag_wd=False):
    """Calculate loss and gradients.
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat.
        y: A Tensor. A batch of labels 
            without sequential_slice and sequential_concat.
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        duration: An int. Num of frames in a sequence.
        param_multiplet_loss: A float. Loss weight.
        param_llr_loss: A float. Loss weight.
        param_wd: A float. Loss weight.
        flag_wd: A boolean. Weight decay or not.
    Returns:
        gradients: A Tensor.
        losses: A list of loss Tensors; namely,
            total_loss: A scalar Tensor that represents the weighted total loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
        logits_concat: A logit Tensor with shape 
            (batch, (duration - order_sprt), order_sprt + 1, num classes). 
            This is the output of 
            datasets.data_processing.sequential_concat(logit_slice, labels_slice).
    """
    x_slice, y_slice = sequential_slice(x, y, order_sprt)

    with tf.GradientTape() as tape:
        logits_slice = model(x_slice, training)
        
        multiplet_loss, llr_loss, logits_concat =  \
            binary_sequential_loss_func(logits_slice, y_slice, training, duration)
        
        total_loss = param_multiplet_loss * multiplet_loss
        total_loss += param_llr_loss * llr_loss

        if flag_wd:
            for variables in model.trainable_variables:
                total_loss += param_wd * tf.nn.l2_loss(variables)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    losses = [total_loss, multiplet_loss, llr_loss]

    return gradients, losses, logits_concat


def get_loss_lstm(model, x, y, training, order_sprt, duration=20):
    """Calculate loss for validation 
       (see the training loop in the main funciton).
    Args:
        model: A tf.keras.Model object.
        x: A Tensor. A batch of time-series input data 
            without sequential_slice and sequential_concat. 
            Shape=(batch, duration, feat dim).
        y: A Tensor. A batch of labels without sequential_slice and
            sequential_concat. Shape = (batch,).
        training: A boolean. Training flag.
        order_sprt: An int. The order of the SPRT.
        duration: An int. Num of frames.
    Returns:
        losses: A list of loss Tensors; namely,
            sum_loss: A scalar Tensor. 
                No multiplication by param_multiplet_loss and param_llr_loss.
            multiplet_loss: A scalar Tensor.
            llr_loss: A scalar Tensor.
    Remark:
        - Note that 
          `sum_loss` in `losses` in this function is different from
          `total_loss` in `losses` in get_gradient_lstm().
    """
    x_slice, y_slice = sequential_slice(x, y, order_sprt)
    logits_slice = model(x_slice, training)    
    multiplet_loss, llr_loss, logits_concat =  \
        binary_sequential_loss_func(logits_slice, y_slice, training, duration)
    sum_loss = multiplet_loss
    sum_loss += llr_loss
    losses = [sum_loss, multiplet_loss, llr_loss]

    return losses, logits_concat


def get_loss_fe(model, x, y, flag_wd, training, calc_grad, param_wd):
    """
    Args:
        model: A tf.keras.Model object.
        x: A Tensor with shape=(batch, H, W, C).
        y: A Tensor with shape (batch,).
        flag_wd: A boolean, whether to decay weight here.
        training: A boolean, the training flag.
        calc_grad: A boolean, whether to calculate gradient.
    """
    if calc_grad:
        with tf.GradientTape() as tape:
            logits, bottleneck_feat = model(x, training)
                # (batch, 2) and (batch, final_size)

            xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=logits))
            
            total_loss = xent

            if flag_wd:
                for variables in model.trainable_variables:
                    total_loss += param_wd * tf.nn.l2_loss(variables)

        gradients = tape.gradient(total_loss, model.trainable_variables)
        losses = [total_loss, xent]

    else:
        logits, bottleneck_feat = model(x, training)
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))

        gradients = None 
        losses = [0., xent]

    return gradients, losses, logits, bottleneck_feat
