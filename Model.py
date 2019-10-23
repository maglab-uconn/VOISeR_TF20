import tensorflow as tf
import numpy as np
from threading import Thread
import time, os, sys, json, argparse, pickle
from collections import Counter

from ProgressBar import progress

from Feeder import Feeder
from Feeder import Load_Data as Load_Variable
import Modules
import Analyzer

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

variable_Dict = Load_Variable()

class VOISeR:
    def __init__(self, start_Epoch, max_Epoch, export_Path):        
        self.feeder = Feeder(start_Epoch, max_Epoch)
        self.export_Path = export_Path

        self.Model_Generate()

    def Model_Generate(self):
        self.layer_Dict = {}
        self.layer_Dict['Input'] = tf.keras.layers.Input(shape=[self.feeder.orthography_Size], dtype= tf.int32)
        self.layer_Dict['Embedding'] = Modules.Embedding(input_dim= len(self.feeder.letter_List))(self.layer_Dict['Input'], self.feeder.max_Pronunciation_Length)
        self.layer_Dict['RNN'] = Modules.RNN(projection_Size= self.feeder.phonology_Size)(self.layer_Dict['Embedding'])[:2]
        self.model = tf.keras.Model(inputs=self.layer_Dict['Input'], outputs= self.layer_Dict['RNN'])

        self.model.summary()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate= hp_Dict['Train']['Learning_Rate']
            )

        self.hidden_Analyzer = Analyzer.Hidden_Analyzer()
        self.output_Analyzer = Analyzer.Ouptut_Analyzer(self.feeder.word_Labels, self.feeder.phoneme_Labels)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, variable_Dict['Max_Word_Length']], dtype=tf.int32),
            tf.TensorSpec(shape=[None, variable_Dict['Max_Pronunciation_Length'], variable_Dict['Phonology_Size']], dtype=tf.float32)
            ],
        autograph= False,
        experimental_relax_shapes= True
        )
    def Train_Step(self, orthographies, pronunciations):
        with tf.GradientTape() as tape:
            _, outputs = self.model(orthographies, training= True)
            if not hp_Dict['Phoneme_Feature_File_Path'] is None:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels= pronunciations,
                    logits= outputs
                    ))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels= pronunciations,
                    logits= outputs
                    ))
        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, variable_Dict['Max_Word_Length']], dtype=tf.int32)
            ],
        autograph= False,
        experimental_relax_shapes= True
        )
    def Inference_Step(self, orthographies):
        hiddens, outputs = self.model(orthographies, training= False)
        if not hp_Dict['Phoneme_Feature_File_Path'] is None:
            outputs = tf.nn.sigmoid(outputs)
        else:
            outputs = tf.nn.softmax(outputs)

        return hiddens, outputs
        
    def Restore(self, start_Epoch):
        if start_Epoch == 0:
            return

        checkpoint_File_Path = os.path.join(self.export_Path, 'Checkpoint', 'E_{}.CHECKPOINT.H5'.format(start_Epoch)).replace('\\', '/')
        
        if not os.path.exists('{}.index'.format(checkpoint_File_Path)):            
            raise Exception('There is no checkpoint about epoch \'{}\'.'.format(start_Epoch))

        self.model.load_weights(checkpoint_File_Path)
        print('Checkpoint \'{}\' is loaded.'.format(checkpoint_File_Path))

    def Train(self):
        os.makedirs(os.path.join(self.export_Path, 'Checkpoint'), exist_ok= True)

        current_Epoch = self.feeder.start_Epoch
        current_Step_in_Epoch = 0
        while not self.feeder.is_Finished or len(self.feeder.pattern_Queue) > 0:
            current_Epoch, is_New_Epoch, orthography_Pattern, phonology_Pattern = self.feeder.Get_Pattern()

            if is_New_Epoch:
                current_Step_in_Epoch = 0
            if is_New_Epoch and current_Epoch % hp_Dict['Train']['Checkpoint_Save_Timing'] == 0:
                self.model.save_weights(os.path.join(self.export_Path, 'Checkpoint', 'E_{}.CHECKPOINT.H5'.format(current_Epoch)).replace('\\', '/'))
                print('Epoch {} checkpoint saved'.format(current_Epoch))
            if is_New_Epoch and current_Epoch % hp_Dict['Train']['Inference_Timing'] == 0:
                self.Inference(epoch= current_Epoch, export_Raw= True)
                        
            start_Time = time.time()
            loss = self.Train_Step(
                orthographies= orthography_Pattern,
                pronunciations= phonology_Pattern
                )

            display_List = [
                'Time: {:0.3f}'.format(time.time() - start_Time),
                'Epoch: {}'.format(current_Epoch),
                'Step in Epoch: {}'.format(current_Step_in_Epoch),
                'Loss: {:0.5f}'.format(loss),
                #'Analyer running...' if any([extract_Thread.is_alive() for extract_Thread in extract_Thread_List]) else ''
                ]
            print('\t\t'.join(display_List))

            current_Step_in_Epoch += 1

        self.model.save_weights(os.path.join(self.export_Path, 'Checkpoint', 'E_{}.CHECKPOINT.H5'.format(current_Epoch + 1)).replace('\\', '/'))
        print('Epoch {} checkpoint saved'.format(current_Epoch + 1))
        self.Inference(epoch= current_Epoch + 1, export_Raw= True)

    def Inference(self, epoch= None, letter_String_List= None, added_Pronunciation_Dict = {}, export_Raw= False):
        os.makedirs(os.path.join(self.export_Path, 'Inference'), exist_ok= True)

        if letter_String_List is None:
            index_Word_Dict = {index: word for word, index in self.feeder.word_Index_Dict.items()}
            letter_String_List = [index_Word_Dict[index] for index in range(len(index_Word_Dict))]

        pattern_List, word_Label_Indices, phoneme_Label_Indices = self.feeder.Get_Inference_Pattern(letter_String_List= letter_String_List, added_Pronunciation_Dict= added_Pronunciation_Dict)

        hiddens_List = []
        outputs_List = []
        for orthography_Pattern in pattern_List:
            hiddens, outputs = self.Inference_Step(
                orthographies= orthography_Pattern
                )
            hiddens_List.append(hiddens)
            outputs_List.append(outputs)

        hiddens = np.vstack(hiddens_List)
        outputs = np.vstack(outputs_List)

        self.Export_Inference(
            letter_String_List,
            hiddens,
            outputs,
            word_Label_Indices,
            phoneme_Label_Indices,
            epoch,
            {key: value for key, value in self.feeder.trained_Pattern_Index_Dict.items() if not epoch is None and key < epoch},
            added_Pronunciation_Dict,
            export_Raw
            )

    def Export_Inference(self, letter_String_List, hiddens, outputs, word_Label_Indices, phoneme_Label_Indices, epoch, trained_Pattern_Index_Dict, added_Pronunciation_Dict = {}, export_Raw= False):
        trained_Pattern_Count_Dict = {index: 0 for index in self.feeder.word_Index_Dict.values()}
        for index_List in trained_Pattern_Index_Dict.values():
            for index in index_List:
                trained_Pattern_Count_Dict[index] += 1

        if export_Raw:
            export_Dict = {
                'Epoch': epoch,
                'Letter_String_List': letter_String_List,
                'Hidden': hiddens,
                'Output': outputs,
                'Trained_Pattern_Count_Dict': trained_Pattern_Count_Dict
                }
            with open(os.path.join(self.export_Path, 'Inference', '{}.Raw.pickle'.format('E_{}'.format(epoch) if not epoch is None else 'Inference')), 'wb') as f:
                pickle.dump(export_Dict, f, protocol= 4)
        
        pattern_Index_List = list(range(hiddens.shape[0]))
        pattern_Index_Batch_List = [pattern_Index_List[x:x+hp_Dict['Analyzer']['Batch_Size']] for x in range(0, len(pattern_Index_List), hp_Dict['Analyzer']['Batch_Size'])]        
        
        hidden_Dict = {}
        result_Dict = {}        
        for index, pattern_Index_Batch in enumerate(pattern_Index_Batch_List):
            batch_Hidden_Dict = self.hidden_Analyzer(inputs= hiddens[pattern_Index_Batch])
            for key, value in batch_Hidden_Dict.items():
                if not key in hidden_Dict.keys():
                    hidden_Dict[key] = []
                hidden_Dict[key].append(value.numpy())
            
            batch_Result_Dict = self.output_Analyzer(inputs= outputs[pattern_Index_Batch], word_label_indices= word_Label_Indices[pattern_Index_Batch], phoneme_label_indices= phoneme_Label_Indices[pattern_Index_Batch])
            for key, value in batch_Result_Dict.items():
                if not key in result_Dict.keys():
                    result_Dict[key] = []
                result_Dict[key].append(value.numpy())
            progress(index + 1, len(pattern_Index_Batch_List), status='Inference analyzer running')
        print()

        hidden_Dict = {key: np.hstack(value_List) if len(value_List[0].shape)== 1 else np.vstack(value_List) for key, value_List in hidden_Dict.items()}
        result_Dict = {key: np.hstack(value_List) if len(value_List[0].shape)== 1 else np.vstack(value_List) for key, value_List in result_Dict.items()}

        index_Phoneme_Dict = {index: phoneme for phoneme, index in self.feeder.phoneme_Index_Dict.items() if type(phoneme) == str}

        column_Title_List = [
            'Epoch',
            'Ortho',
            'Phono',
            'Length',
            'Probability',
            'MeanRT',
            'Trained_Count',
            'Cosine_Similarity',
            'Mean_Squared_Error',
            'Euclidean_Distance',
            'Cross_Entropy',
            'Exported_Pronunciation',
            'Accuracy_Max_CS',
            'Accuracy_Min_MSE',
            'Accuracy_Min_ED',
            'Accuracy_Min_CE',
            'Accuracy_Pronunciation',
            'Hidden_Cosine_Similarity',
            'Hidden_Mean_Squared_Error',
            'Hidden_Euclidean_Distance',
            'Hidden_Cross_Entropy',
            ]
        
        export_List = ['\t'.join(column_Title_List)]
        for index, (letter_String, rt_CS, rt_MSE, rt_ED, rt_CE, exported_Pronunciation, acc_CS, acc_MSE, acc_ED, acc_CE, acc_Pronunciation, hidden_CS, hidden_MSE, hidden_ED, hidden_CE) in enumerate(zip(
            letter_String_List,
            result_Dict['RT', 'CS'],
            result_Dict['RT', 'MSE'],
            result_Dict['RT', 'ED'],
            result_Dict['RT', 'CE'],
            result_Dict['Export', 'Pronunciation'],
            result_Dict['ACC', 'Max_CS'],
            result_Dict['ACC', 'Min_MSE'],
            result_Dict['ACC', 'Min_ED'],
            result_Dict['ACC', 'Min_CE'],
            result_Dict['ACC', 'Pronunciation'],
            hidden_Dict['CS'],
            hidden_Dict['MSE'],
            hidden_Dict['ED'],
            hidden_Dict['CE'],
            )):
            is_Word = letter_String in self.feeder.word_Index_Dict.keys()
            new_Line_List = [
                str(epoch or 'None'),
                letter_String,
                added_Pronunciation_Dict[letter_String] if letter_String in added_Pronunciation_Dict.keys() else self.feeder.pronunciation_Dict[letter_String],
                str(len(letter_String)),
                str(self.feeder.frequency_Dict[letter_String]) if is_Word else 'None',
                str(self.feeder.human_RT_Dict[letter_String]) if is_Word else 'None',
                str(trained_Pattern_Count_Dict[index]) if is_Word else 'None',
                str(rt_CS),
                str(rt_MSE),
                str(rt_ED),
                str(rt_CE),
                ''.join([index_Phoneme_Dict[phoneme_Index] for phoneme_Index in exported_Pronunciation]),
                str(acc_CS),
                str(acc_MSE),
                str(acc_ED),
                str(acc_CE),
                str(acc_Pronunciation),
                str(hidden_CS),
                str(hidden_MSE),
                str(hidden_ED),
                str(hidden_CE),
                ]
            export_List.append('\t'.join(new_Line_List))

        with open(os.path.join(self.export_Path, 'Inference', '{}.Summary.txt'.format('E_{}'.format(epoch) if not epoch is None else 'Inference')), 'w') as f:
            f.write('\n'.join(export_List))

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-se", "--start_epoch", required=False)
    argParser.add_argument("-e", "--epoch", required=True)
    argParser.add_argument("-idx", "--idx", required=False)
    argument_Dict = vars(argParser.parse_args())

    extract_Path_List = []
    if hp_Dict['RNN']['Use_Feedback'] and hp_Dict['RNN']['Use_Recurrent']:
        hidden_Calc_Type = 'B'
    if hp_Dict['RNN']['Use_Feedback'] and not hp_Dict['RNN']['Use_Recurrent']:
        hidden_Calc_Type = 'O'
    if not hp_Dict['RNN']['Use_Feedback'] and hp_Dict['RNN']['Use_Recurrent']:
        hidden_Calc_Type = 'H'
    extract_Path_List.append('HT_{}'.format(hidden_Calc_Type))
    extract_Path_List.append("HU_{}".format(hp_Dict['RNN']['Size']))
    extract_Path_List.append("LR_{}".format(str(hp_Dict['Train']['Learning_Rate'])[2:]))
    extract_Path_List.append("E_{}".format(int(argument_Dict['epoch'])))
    extract_Path_List.append("TT_{}".format(hp_Dict['Train']['Inference_Timing']))
    if hp_Dict['Use_Frequency']:
        extract_Path_List.append("Fre")
    if not hp_Dict['Orthography_Embedding_Size'] is None:
        extract_Path_List.append("EMB_{}".format(hp_Dict['Orthography_Embedding_Size']))
    if not hp_Dict['Phoneme_Feature_File_Path'] is None:
        extract_Path_List.append("DSTR_True")
    if argument_Dict['idx'] is not None:
        extract_Path_List.append("IDX_{}".format(argument_Dict['idx']))
    extract_Path = os.path.join(hp_Dict['Export_Path'], ".".join(extract_Path_List))

    new_VOISeR = VOISeR(
        start_Epoch= int(argument_Dict['start_epoch'] or 0),
        max_Epoch= int(argument_Dict['epoch']),
        export_Path= extract_Path
        )
    new_VOISeR.Restore(start_Epoch= int(argument_Dict['start_epoch'] or 0))
    new_VOISeR.Train()