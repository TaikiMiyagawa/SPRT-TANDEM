"""LSTM model definition compatible with TensorFlow's eager execution.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation


class LSTMModelLite(tf.keras.Model):
    """LSTM model with TF2.0.0 implementation.
    Caution:
        If you are to use the N-th-order SPRT, 
        inputs argument in `call` must have shape (batch, N, feature dimension),
        not (batch, duration (e.g., duration=20 for nosaic MNIST), feature dimension),
        because the memory state is deleted after the inputs.shape[1]-th time step.
        Note that the reshape (batch, duration, feat dim) -> (batch*(duration-N+1), N, feat dim) can
        be performed with datasets.data_processing.sequential_slice_nosaic_mnist().
    Remark: 
        `Lite` has no meanings, just ignore.
    """
    def __init__(self, nb_cls, width_lstm, dropout=0., activation="tanh"):
        """
        Args:
            nb_cls: An int. The dimension of the output logit vectors.
            width_lstm: An int. The width of LSTM hidden fc layer.
            dropout: An float in [0, 1]. Dropout rate, not keep_prob.
            activation: A string. For activation argument in tf.keras.layers.LSTM.
                Note that
                recurrent_activation (i.e., input, output, and forget gate activation) 
                in tf.keras.layers.LSTM is "sigmoid" by default and is fixed.
        """
        super(LSTMModelLite, self).__init__(name="PeepholeLSTM_Lite")

        # Parameters
        self.nb_cls = nb_cls
        self.width_lstm = width_lstm
        self.dropout = dropout
        self.activation = activation
        
        ## Feature extraction fully-connected layer
        #self.fc_featext = Dense(self.width_lstm, activation=Activation(self.activation), use_bias=True)
        #self.bn_featext = BatchNormalization()
        #self.activation_featext = Activation(self.activation)

        # LSTM cell
        self.lstm_cell = tf.keras.experimental.PeepholeLSTMCell(
            units=self.width_lstm,
            activation=Activation(self.activation),
            unit_forget_bias=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout)

        # RNN
        self.rnn = tf.keras.layers.RNN(
            self.lstm_cell,
            return_sequences=True,
            return_state=True)

        # Logit generation fully-connected layer
        self.bn_logit = BatchNormalization()
        self.activation_logit = Activation(self.activation)
        self.fc_logit = Dense(nb_cls, activation=None, use_bias=False)
    
    #def fc_bn_act_featext(self, x, training):
    #    """
    #    Args:
    #        x: A Tensor. Input feature with shape=(batch*duration, 784) for nosaic MNIST.
    #    Return:
    #        x: A Tensor. Logit with shape=(batch*duration, self.width_lstm)        
    #    """
    #    x = self.fc_featext(x)
    #    x = self.bn_featext(x, training=training)
    #    x = self.activation_featext(x)
    #    return x
        
    def bn_act_fc_logit(self, x, training):
        """
        Args:
            x: A Tensor. Output of LSTM with shape=(batch*duration, self.width_lstm).
        Return:
            x: A Tensor. Logit with shape=(batch*duration, self.nb_cls)        
        """
        x = self.bn_logit(x, training=training)
        x = self.activation_logit(x)
        x = self.fc_logit(x)
        return x

    def call(self, inputs, training):
        """Calc logits.
        Args:
            inputs: A Tensor with shape=(batch, duration, feature dimension). E.g. (128, 20, 784) for nosaic MNIST.
            training: A boolean. Training flag used in BatchNormalization and dropout.
        Returns:
            outputs: A Tensor with shape=(batch, duration, nb_cls).
        """
        # Parameters
        inputs_shape = inputs.shape 
        duration = inputs_shape[1] # 20 by default for nosaic MNIST

        ## Feature extraction
        #inputs_featext = tf.reshape(inputs, (-1,784)) 
        #    # (B, T, 784) -> (BT, 784)
        #inputs_featext = self.fc_bn_act_featext(
        #    inputs_featext, 
        #    training=training) 
        #    # (BT, 784) -> (BT, self.width.lstm)
        #inputs_featext = tf.reshape(
        #    inputs_featext, 
        #    (-1, duration, self.width_lstm)) 
        #    # (BT, self.width_lstm) -> (B, T, self.width_lstm)

        # Feedforward
        #outputs, _, _ = self.rnn(inputs_featext, training=training)
        outputs, _, _ = self.rnn(inputs, training=training)

        # Make logits
        outputs = tf.reshape(outputs, (-1, self.width_lstm))
        outputs = self.bn_act_fc_logit(outputs, training=training)
        outputs = tf.reshape(outputs, (-1, duration, self.nb_cls)) # (B, T, nb_cls)

        return outputs # A Tensor with shape=(batch, duration, nb_cls)