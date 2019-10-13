import tensorflow as tf
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Embedding(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Embedding, self).__init__(name= '')
        self.input_dim = input_dim
        if not hp_Dict['Orthography_Embedding_Size'] is None:
            self.embedding = tf.keras.layers.Embedding(
                input_dim= input_dim,
                output_dim= hp_Dict['Orthography_Embedding_Size']
                )

    def call(self, inputs, decoder_length):
        if not hp_Dict['Orthography_Embedding_Size'] is None:
            new_Tensor = self.embedding(inputs)

        else:
            new_Tensor = tf.one_hot(
                indices= inputs,
                depth= self.input_dim,
                )

        batch_Size = tf.shape(new_Tensor)[0]
        time, dim = new_Tensor.shape[1:3]
        new_Tensor = tf.reshape(new_Tensor, (batch_Size, time * dim))
        new_Tensor = tf.tile(tf.expand_dims(new_Tensor, axis=1), multiples=[1, decoder_length ,1])

        return new_Tensor


class RNN(tf.keras.Model):
    def __init__(self, projection_Size):
        super(RNN, self).__init__(name= '')
        self.cell = Cell(
            units= hp_Dict['RNN']['Size'],
            projection_units= projection_Size,
            use_feedback= hp_Dict['RNN']['Use_Feedback'],
            use_recurrent= hp_Dict['RNN']['Use_Recurrent'],
            projection_activation= 'sigmoid' if not hp_Dict['Phoneme_Feature_File_Path'] is None else 'softmax'
            )

    @tf.function
    def call(self, inputs, training= False):
        input_Data = tf.transpose(inputs, [1, 0, 2])    #[Batch, Time, Dim] -> [Time, Batch, Dim]
        outputs = tf.TensorArray(input_Data.dtype, input_Data.shape[0])
        projections = tf.TensorArray(input_Data.dtype, input_Data.shape[0])        

        state = self.cell.get_initial_state(batch_size= tf.shape(input_Data)[1], dtype= input_Data.dtype)
        projection = self.cell.get_initial_projection(batch_size= tf.shape(input_Data)[1], dtype= input_Data.dtype)
        
        for index in tf.range(input_Data.shape[0]):
            output, projection, state = self.cell(input_Data[index], projection, state, training)
            outputs = outputs.write(index, output)
            projections = projections.write(index, projection)

        return tf.transpose(outputs.stack(), [1, 0, 2]), tf.transpose(projections.stack(), [1, 0, 2]), state


from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils, tf_utils
from tensorflow.python.ops import array_ops, control_flow_util, state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers import recurrent
class Cell(recurrent.DropoutRNNCellMixin, Layer):
    """Cell class for VOISeR. 
    This class is referred to SimpleRNN of TF 2.0:
        https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/recurrent.py#L1109-L1268
    Arguments:
        units: Positive integer, dimensionality of the output space.
        projection_units: Positive integer, dimensionality of the projection space.
        use_feedback: Boolean, use projection state.
            Default: True
        use_recurrent: Boolean, use recurrent state.
            Default: True
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        projection_activation: Activation function to use. This is only used in recurrent. It does not apply to the finally exported projection.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        use_bias: Boolean, whether the projection uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        projection_initializer: Initializer for the `projection_kernel`
            weights matrix, used for the linear transformation of the hidden.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent state.
        feedback_initializer: Initializer for the `feedback_kernel`
            weights matrix, used for the linear transformation of the feedback state.            
        bias_initializer: Initializer for the bias vector.
        projection_bias_initializer: Initializer for the projection bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        projection_regularizer: Regularizer function applied to
            the `projection_kernel` weights matrix.
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
        feedback_regularizer: Regularizer function applied to
            the `feedback_kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        projection_bias_regularizer: Regularizer function applied to the projection bias vector.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        projection_constraint: Constraint function applied to
            the `projection_kernel` weights matrix.
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
        feedback_constraint: Constraint function applied to
            the `feedback_kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
        projection_bias_constraint: Constraint function applied to the bias vector.
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    Call arguments:
        inputs: A 2D tensor.
        states: List of state tensors corresponding to the previous timestep.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    """

    def __init__(
        self,
        units,
        projection_units,
        use_feedback= True,
        use_recurrent= True,
        activation='tanh',
        projection_activation='sigmoid',
        use_bias=True,
        use_projection_bias=True,
        kernel_initializer='glorot_uniform',
        projection_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        feedback_initializer='orthogonal',
        bias_initializer='zeros',
        projection_bias_initializer='zeros',
        kernel_regularizer=None,
        projection_regularizer=None,
        recurrent_regularizer=None,
        feedback_regularizer=None,
        bias_regularizer=None,
        projection_bias_regularizer=None,
        kernel_constraint=None,
        projection_constraint=None,
        recurrent_constraint=None,
        feedback_constraint=None,
        bias_constraint=None,
        projection_bias_constraint=None,
        dropout=0.,
        recurrent_dropout=0.,
        **kwargs
        ):
        super(Cell, self).__init__(**kwargs)
        self.units = units
        self.projection_units = projection_units
        self.use_feedback = use_feedback
        self.use_recurrent = use_recurrent
        self.activation = activations.get(activation)
        self.projection_activation = activations.get(projection_activation)
        self.use_bias = use_bias
        self.use_projection_bias = use_projection_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.projection_initializer = initializers.get(projection_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.feedback_initializer = initializers.get(feedback_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.projection_bias_initializer = initializers.get(projection_bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.projection_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.feedback_regularizer = regularizers.get(feedback_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.projection_bias_regularizer = regularizers.get(projection_bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.projection_constraint = constraints.get(projection_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.feedback_constraint = constraints.get(feedback_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.projection_bias_constraint = constraints.get(projection_bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        self.state_size = self.units
        self.output_size = self.units
        self.projection_size = self.projection_units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        self.projection_kernel = self.add_weight(
            shape=(self.units, self.projection_units),
            name='projection_kernel',
            initializer=self.projection_initializer,
            regularizer=self.projection_regularizer,
            constraint=self.projection_constraint)

        if self.use_recurrent:
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                name='recurrent_kernel',
                initializer=self.recurrent_initializer,
                regularizer=self.recurrent_regularizer,
                constraint=self.recurrent_constraint)

        if self.use_feedback:
            self.feedback_kernel = self.add_weight(
                shape=(self.projection_units, self.units),
                name='feedback_kernel',
                initializer=self.feedback_initializer,
                regularizer=self.feedback_regularizer,
                constraint=self.feedback_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                shape=(self.projection_units,),
                name='projection_bias',
                initializer=self.projection_bias_initializer,
                regularizer=self.projection_bias_regularizer,
                constraint=self.projection_bias_constraint)
        else:
            self.projection_bias = None

        self.built = True

    def call(self, inputs, prev_projection, states, training=None):
        prev_output = states[0]

        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)

        if dp_mask is not None:
            inputs = inputs * dp_mask        
        output = K.dot(inputs, self.kernel)
        
        if self.use_recurrent:
            if rec_dp_mask is not None:
                prev_output = prev_output * rec_dp_mask
            output += K.dot(prev_output, self.recurrent_kernel)

        if self.use_feedback:
            if self.projection_activation is not None:
                prev_projection = self.projection_activation(prev_projection)
            output += K.dot(prev_projection, self.feedback_kernel)

        if self.bias is not None:
            output = K.bias_add(output, self.bias)
            
        if self.activation is not None:
            output = self.activation(output)
        
        projection = K.dot(output, self.projection_kernel)        

        if self.projection_bias is not None:
            projection = K.bias_add(projection, self.projection_bias)
        
        return output, projection, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [recurrent._generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)]

    def get_initial_projection(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = array_ops.shape(inputs)[0]
            dtype = inputs.dtype
        return array_ops.zeros([batch_size, self.projection_size], dtype=dtype)

    def get_config(self):        
        config = {
            'units':
                self.units,
            'projection_units':
                self.projection_units,
            'use_feedback':
                self.use_feedback,
            'use_recurrent':
                self.use_recurrent,
            'activation':
                activations.serialize(self.activation),
            'projection_activation':
                activations.serialize(self.projection_activation),
            'use_bias':
                self.use_bias,
            'use_projection_bias':
                self.use_projection_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'projection_initializer':
                initializers.serialize(self.projection_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'recurrent_initializer':
                initializers.serialize(self.feedback_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'bias_initializer':
                initializers.serialize(self.projection_bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'projection_regularizer':
                regularizers.serialize(self.projection_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'feedback_regularizer':
                regularizers.serialize(self.feedback_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'projection_bias_regularizer':
                regularizers.serialize(self.projection_bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'projection_constraint':
                constraints.serialize(self.projection_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'feedback_constraint':
                constraints.serialize(self.feedback_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'projection_bias_constraint':
                constraints.serialize(self.projection_bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(Cell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))