import tensorflow as tf
import json

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Ouptut_Analyzer(tf.keras.layers.Layer):
    def __init__(self, all_word_labels, all_phoneme_labels):
        '''
        all_word_labels: [37610, 13, 18]
        all_phoneme_labels: [42, 18]
        '''
        super(Ouptut_Analyzer, self).__init__(name= '')

        self.all_word_labels = tf.Variable(all_word_labels, trainable= False, dtype= tf.float32)
        self.all_phoneme_labels = tf.Variable(all_phoneme_labels, trainable= False, dtype= tf.float32)

    @tf.function
    def call(self, inputs, word_label_indices, phoneme_label_indices):
        '''
        inputs(outputs): [Batch, 13, 18]
        word_label_indices: [Batch]
        phoneme_label_indices: [Batch, 13]
        '''
        outputs= inputs
        reshaped_Outputs = tf.reshape(outputs, [outputs.shape[0], outputs.shape[1] * outputs.shape[2]])  #[Batch, 234]
        reshaped_Word_Labels = tf.reshape(self.all_word_labels, [self.all_word_labels.shape[0], outputs.shape[1] * outputs.shape[2]])   #[37610, 234]

        tiled_Reshaped_Outputs = tf.tile(tf.expand_dims(reshaped_Outputs, axis=1), multiples= [1, reshaped_Word_Labels.shape[0], 1])  #[Batch, 37610, 234]
        tiled_Reshaped_Word_Labels = tf.tile(tf.expand_dims(reshaped_Word_Labels, axis=0), multiples= [reshaped_Outputs.shape[0], 1, 1]) #[Batch, 37610, 234]

        cosine_Similarity = tf.reduce_sum(tiled_Reshaped_Word_Labels * tiled_Reshaped_Outputs, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Word_Labels, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Outputs, 2), axis = 2)))  #[batch, 37610]        
        mean_Squared_Error = tf.reduce_mean(tf.pow(tiled_Reshaped_Word_Labels - tiled_Reshaped_Outputs, 2), axis=2)  #[Batch, 37610]
        euclidean_Distance = tf.sqrt(tf.reduce_sum(tf.pow(tiled_Reshaped_Word_Labels - tiled_Reshaped_Outputs, 2), axis=2))  #[Batch, 37610]
        cross_Entropy = -tf.reduce_mean(tiled_Reshaped_Word_Labels * tf.math.log(tiled_Reshaped_Outputs + 1e-8) + (1 - tiled_Reshaped_Word_Labels) * tf.math.log(1 - tiled_Reshaped_Outputs + 1e-8), axis = 2)  #[Batch, 37610]

        if not hp_Dict['Phoneme_Feature_File_Path'] is None:
            tiled_Outputs = tf.tile(tf.expand_dims(outputs, axis= 2), multiples = [1, 1, self.all_phoneme_labels.shape[0], 1]);   #[Batch, 13, 42, 18]
            tiled_Phoneme_Labels = tf.tile(tf.expand_dims(tf.expand_dims(self.all_phoneme_labels, axis=0), axis=0), multiples = [tf.shape(outputs)[0], tf.shape(outputs)[1], 1, 1]);   #[Batch, 13, 42, 18]
            phoneme_Cosine_Similarity = tf.reduce_sum(tiled_Phoneme_Labels * tiled_Outputs, axis = 3) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Phoneme_Labels, 2), axis = 3)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Outputs, 2), axis = 3)))  #[Batch, 13, 42]
            pattern_Argmax = tf.argmax(phoneme_Cosine_Similarity, axis= 2, output_type=tf.int32);  #[Batch, 13]
        else:
            pattern_Argmax = tf.argmax(outputs, axis= 2, output_type=tf.int32);  #[Batch, 13]

         

        gather_Indices = tf.stack([tf.range(tf.shape(word_label_indices)[0], dtype=tf.int32), word_label_indices], axis=1)
        
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
        hiddens = (inputs + 1) / 2  #tanh(-1, 1) to (0, 1)
        hiddens_Compare = tf.concat([tf.zeros((tf.shape(hiddens)[0], 1, tf.shape(hiddens)[2])), hiddens], axis=1)[:,:-1,:]    #[Batch, 13, 300]

        cosine_Similarity = tf.reduce_sum(tf.reduce_sum(hiddens * hiddens_Compare, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(hiddens, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(hiddens_Compare, 2), axis = 2))), axis=1)  #[Batch]
        mean_Squared_Error = tf.reduce_sum(tf.reduce_mean(tf.pow(hiddens - hiddens_Compare, 2), axis=2), axis=1)  #[Batch]
        euclidean_Distance = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.pow(hiddens - hiddens_Compare, 2), axis=2)), axis=1)  #[Batch]
        cross_Entropy = tf.reduce_sum(-tf.reduce_mean(hiddens * tf.math.log(hiddens_Compare + 1e-8) + (1 - hiddens) * tf.math.log(1 - hiddens_Compare + 1e-8), axis = 2), axis=1)  #[Batch]

        return {
            'CS': cosine_Similarity,  #[Batch]
            'MSE':mean_Squared_Error,  #[Batch]
            'ED': euclidean_Distance,  #[Batch]
            'CE': cross_Entropy,  #[Batch]
            }