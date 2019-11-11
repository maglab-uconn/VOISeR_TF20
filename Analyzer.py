import tensorflow as tf
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Ouptut_Analyzer(tf.keras.layers.Layer):
    def __init__(self, all_word_labels, all_phoneme_labels):
        '''
        all_word_labels: [all trained pattern size, step, phoneme vector size]
        all_phoneme_labels: [all phoneme count, phoneme vector size]
        '''
        super(Ouptut_Analyzer, self).__init__(name= '')

        self.all_word_labels = tf.Variable(all_word_labels, trainable= False, dtype= tf.float32)    #To compare, storing the pronunciation matrix about each word
        self.all_phoneme_labels = tf.Variable(all_phoneme_labels, trainable= False, dtype= tf.float32)  #To compare, storing the phoneme vector about each phoneme

    @tf.function
    def call(self, inputs, word_label_indices, phoneme_label_indices, added_Word_Labels):
        '''
        inputs(outputs): [Batch, 13, 18]
        word_label_indices: [Batch]
        phoneme_label_indices: [Batch, 13]
        '''
        outputs= inputs
        word_Labels = tf.concat([self.all_word_labels, added_Word_Labels], axis=0)
        reshaped_Outputs = tf.reshape(outputs, [outputs.shape[0], outputs.shape[1] * outputs.shape[2]])  #[Batch, step * phoneme vector size]
        reshaped_Word_Labels = tf.reshape(word_Labels, [word_Labels.shape[0], outputs.shape[1] * outputs.shape[2]])   #[all trained pattern size, step * phoneme vector size]

        # Increase dimension and tiled for 2D comparing.
        tiled_Reshaped_Outputs = tf.tile(tf.expand_dims(reshaped_Outputs, axis=1), multiples= [1, reshaped_Word_Labels.shape[0], 1])  #[Batch, all trained pattern size, step * phoneme vector size]
        tiled_Reshaped_Word_Labels = tf.tile(tf.expand_dims(reshaped_Word_Labels, axis=0), multiples= [reshaped_Outputs.shape[0], 1, 1]) #[Batch, all trained pattern size, step * phoneme vector size]

        cosine_Similarity = tf.reduce_sum(tiled_Reshaped_Word_Labels * tiled_Reshaped_Outputs, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Word_Labels, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Outputs, 2), axis = 2)))  #[batch, all trained pattern size]        
        mean_Squared_Error = tf.reduce_mean(tf.pow(tiled_Reshaped_Word_Labels - tiled_Reshaped_Outputs, 2), axis=2)  #[Batch, all trained pattern size]
        euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Word_Labels - tiled_Reshaped_Outputs, 2), axis=2))  #[Batch, all trained pattern size]
        cross_Entropy = -tf.reduce_mean(tiled_Reshaped_Word_Labels * tf.math.log(tiled_Reshaped_Outputs + 1e-8) + (1 - tiled_Reshaped_Word_Labels) * tf.math.log(1 - tiled_Reshaped_Outputs + 1e-8), axis = 2)  #[Batch, all trained pattern size]

        if not hp_Dict['Phoneme_Feature_File_Path'] is None:    # If phoneme representation is feature based, vector must be changed a single value like cosine similarity for comparison between phonemes.
            tiled_Outputs = tf.tile(tf.expand_dims(outputs, axis= 2), multiples = [1, 1, self.all_phoneme_labels.shape[0], 1])   #[Batch, Step, all phoneme count, phoneme vector size]
            tiled_Phoneme_Labels = tf.tile(tf.expand_dims(tf.expand_dims(self.all_phoneme_labels, axis=0), axis=0), multiples = [tf.shape(outputs)[0], tf.shape(outputs)[1], 1, 1])   #[Batch, Step, all phoneme count, phoneme vector size]
            phoneme_Cosine_Similarity = tf.reduce_sum(tiled_Phoneme_Labels * tiled_Outputs, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Phoneme_Labels, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Outputs, 2), axis = 3)))  #[Batch, Step, all phoneme count]
            pattern_Argmax = tf.argmax(phoneme_Cosine_Similarity, axis= 2, output_type=tf.int32)  #[Batch, Step]
        else:   # If phoneme representation is one-hot, just conducting comparison between phonemes activations.
            pattern_Argmax = tf.argmax(outputs, axis= 2, output_type=tf.int32)  #[Batch, 13]

        gather_Indices = tf.stack([tf.range(tf.shape(word_label_indices)[0], dtype=tf.int32), word_label_indices], axis=1)  #To get RT, model just uses target index's value
        
        return {
            ('RT', 'CS'): tf.gather_nd(cosine_Similarity, indices= gather_Indices) ,  #[Batch]
            ('RT', 'MSE'): tf.gather_nd(mean_Squared_Error, indices= gather_Indices) ,  #[Batch]
            ('RT', 'ED'): tf.gather_nd(euclidean_Distance, indices= gather_Indices) ,  #[Batch]
            ('RT', 'CE'): tf.gather_nd(cross_Entropy, indices= gather_Indices) ,  #[Batch]

            ('ACC', 'Max_CS'): tf.equal(word_label_indices, tf.argmax(cosine_Similarity, axis= -1, output_type=tf.int32)),  #[Batch]
            ('ACC', 'Min_MSE'): tf.equal(word_label_indices, tf.argmin(mean_Squared_Error, axis= -1, output_type=tf.int32)), #[Batch]
            ('ACC', 'Min_ED'): tf.equal(word_label_indices, tf.argmin(euclidean_Distance, axis= -1, output_type=tf.int32)), #[Batch]
            ('ACC', 'Min_CE'): tf.equal(word_label_indices, tf.argmin(cross_Entropy, axis= -1, output_type=tf.int32)),  #[Batch]
            ('ACC', 'Pronunciation'): tf.cast(tf.reduce_mean(tf.cast(tf.equal(pattern_Argmax, phoneme_label_indices), dtype=tf.int32), axis=-1), dtype= tf.bool), #[Batch]

            ('Export', 'Pronunciation'): pattern_Argmax #[Batch, 13]
            }

class Hidden_Analyzer(tf.keras.layers.Layer):
    def __init__(self):
        super(Hidden_Analyzer, self).__init__(name= '')

    @tf.function
    def call(self, inputs):
        '''
        inputs(hiddens): [Batch, 13, 300]
        '''
        hiddens = (inputs + 1.0) / 2.0  #tanh(-1, 1) to (0, 1)
        hiddens_Compare = tf.concat([tf.zeros((tf.shape(hiddens)[0], 1, tf.shape(hiddens)[2])), hiddens], axis=1)[:,:-1,:]    #[Batch, Step, hidden size]. Initial step is zero vector, and last step removed for matching step.

        cosine_Similarity = tf.reduce_sum(tf.reduce_sum(hiddens * hiddens_Compare, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(hiddens, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(hiddens_Compare, 2), axis = 2)) + 1e-8), axis=1)  #[Batch]
        mean_Squared_Error = tf.reduce_sum(tf.reduce_mean(tf.pow(hiddens - hiddens_Compare, 2), axis=2), axis=1)  #[Batch]
        euclidean_Distance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(hiddens - hiddens_Compare, 2), axis=2)), axis=1)  #[Batch]
        cross_Entropy = tf.reduce_sum(-tf.reduce_mean(hiddens * tf.math.log(hiddens_Compare + 1e-8) + (1 - hiddens) * tf.math.log(1 - hiddens_Compare + 1e-8), axis = 2), axis=1)  #[Batch]

        return {
            'CS': cosine_Similarity,  #[Batch]
            'MSE':mean_Squared_Error,  #[Batch]
            'ED': euclidean_Distance,  #[Batch]
            'CE': cross_Entropy,  #[Batch]
            }