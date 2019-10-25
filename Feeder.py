import numpy as np;
import tensorflow as tf;
from threading import Thread;
from collections import deque, Sequence;
from random import shuffle;
import time, json;

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

def Load_Data():
    data_Dict = {}

    with open (hp_Dict['Train']['Pattern_File_Path'], 'r') as f:
        readLines = f.readlines()[1:]

    splited_ReadLine = [readLine.strip().split(',')[1:] for readLine in readLines]
    data_Dict['Word_Index_Dict'] = {word.lower(): index for index, (word, _, _, _, _) in enumerate(splited_ReadLine)}
    data_Dict['Pronunciation_Dict'] = {word.lower(): pronunciation for word, pronunciation, _, _, _ in splited_ReadLine}
    data_Dict['Frequency_Dict'] = {word.lower(): float(frequency) * 0.05 + 0.1 for word, _, _, frequency, _ in splited_ReadLine}
    data_Dict['Human_RT_Dict'] = {word.lower(): float(rt) for word, _, _, _, rt in splited_ReadLine}
    
    data_Dict['Max_Word_Length'] = max([len(word) for word in data_Dict['Word_Index_Dict'].keys()])
    data_Dict['Max_Pronunciation_Length'] = max([len(pronunciation) for pronunciation in data_Dict['Pronunciation_Dict'].values()])

    if len(data_Dict['Word_Index_Dict'].keys()) != len(set(data_Dict['Word_Index_Dict'].keys())):
        raise Exception('More than one word of the same spelling is in the data!')

    letter_Set = set()
    phoneme_Set = set()
    for word, pronunciation in data_Dict['Pronunciation_Dict'].items():
        letter_Set.update(set(word))
        phoneme_Set.update(set(pronunciation))
    data_Dict['Letter_List'] = list(sorted(list(letter_Set))) + ['_']
    data_Dict['Phoneme_List'] = list(sorted(list(phoneme_Set))) + ['_']
        
    #There are two key types: [letter], [slot_Index, letter]
    data_Dict['Letter_Index_Dict'] = {letter: letter_Index for letter_Index, letter in enumerate(data_Dict['Letter_List'])}
    for slot_Index in range(data_Dict['Max_Word_Length']):
        for letter_Index, letter in enumerate(data_Dict['Letter_List']):
            data_Dict['Letter_Index_Dict'][slot_Index, letter] = slot_Index * len(data_Dict['Letter_List']) + letter_Index

    #There are two key types: [phoneme], [slot_Index, phoneme]
    data_Dict['Phoneme_Index_Dict'] = {phoneme: phoneme_Index for phoneme_Index, phoneme in enumerate(data_Dict['Phoneme_List'])}
    for slot_Index in range(data_Dict['Max_Pronunciation_Length']):
        for phoneme_Index, phoneme in enumerate(data_Dict['Phoneme_List']):
            data_Dict['Phoneme_Index_Dict'][slot_Index, phoneme] = slot_Index * len(data_Dict['Phoneme_List']) + phoneme_Index

    if not hp_Dict['Phoneme_Feature_File_Path'] is None:
        with open (hp_Dict['Phoneme_Feature_File_Path'], 'r') as f:
            readLines = f.readlines()[1:]
        splited_ReadLine = [readLine.strip().split(',') for readLine in readLines]
        data_Dict['Phoneme_Pattern_Dict'] = {pattern[0]: np.array([float(x) for x in pattern[1:]], dtype=np.float32) for pattern in splited_ReadLine}
    else:
        data_Dict['Phoneme_Pattern_Dict'] = {}
        for phoneme in data_Dict['Phoneme_List']:
            new_Pattern = np.zeros(shape = len(data_Dict['Phoneme_List']))
            new_Pattern[data_Dict['Phoneme_Index_Dict'][phoneme]] = 1
            data_Dict['Phoneme_Pattern_Dict'][phoneme] = new_Pattern
                    
    data_Dict['Orthography_Size'] = data_Dict['Max_Word_Length']
    data_Dict['Phonology_Size'] = data_Dict['Phoneme_Pattern_Dict'][data_Dict['Phoneme_List'][0]].shape[0]

    return data_Dict

class Feeder:
    def __init__(self, start_Epoch, max_Epoch):
        self.start_Epoch = start_Epoch
        self.max_Epoch = max_Epoch
        
        self.Load_Data()
        self.Analyzer_Label_Generate()

        self.is_Finished = False
        self.pattern_Queue = deque()
        self.trained_Pattern_Index_Dict = {}
                
        pattern_Generate_Thread = Thread(target=self.Pattern_Generate)
        pattern_Generate_Thread.daemon = True
        pattern_Generate_Thread.start()
    
    def Load_Data(self):
        data_Dict = Load_Data()

        self.word_Index_Dict = data_Dict['Word_Index_Dict']
        self.pronunciation_Dict = data_Dict['Pronunciation_Dict']
        self.frequency_Dict = data_Dict['Frequency_Dict']
        self.human_RT_Dict = data_Dict['Human_RT_Dict']
        
        self.max_Word_Length = data_Dict['Max_Word_Length']
        self.max_Pronunciation_Length = data_Dict['Max_Pronunciation_Length']

        self.letter_List = data_Dict['Letter_List']
        self.phoneme_List = data_Dict['Phoneme_List']
            
        #There are two key types: [letter], [slot_Index, letter]
        self.letter_Index_Dict = data_Dict['Letter_Index_Dict']

        #There are two key types: [phoneme], [slot_Index, phoneme]
        self.phoneme_Index_Dict = data_Dict['Phoneme_Index_Dict']

        self.phoneme_Pattern_Dict = data_Dict['Phoneme_Pattern_Dict']
                        
        self.orthography_Size = data_Dict['Orthography_Size']
        self.phonology_Size = data_Dict['Phonology_Size']
        
    def Analyzer_Label_Generate(self):
        index_Word_Dict = {index: word for word, index in self.word_Index_Dict.items()}

        self.word_Labels = np.stack([
            self.Pronunciation_to_Pattern(self.pronunciation_Dict[index_Word_Dict[index]])
            for index in range(len(index_Word_Dict))
            ])


        self.phoneme_Labels = np.vstack([self.phoneme_Pattern_Dict[phoneme] for phoneme in self.phoneme_List])
    
    def Word_to_Pattern(self, word):
        word = word + '_' * (self.max_Word_Length - len(word))

        return np.array([self.letter_Index_Dict[letter] for letter in word], dtype=np.int32)

    def Pronunciation_to_Pattern(self, pronunciation):
        pronunciation = pronunciation + '_' * (self.max_Pronunciation_Length - len(pronunciation))

        new_Pattern = np.zeros((self.max_Pronunciation_Length, self.phonology_Size))
        for index, phoneme in enumerate(pronunciation):
            new_Pattern[index] = self.phoneme_Pattern_Dict[phoneme]
                    
        return new_Pattern

    def Pattern_Generate(self):
        #Batched Pattern Making
        pattern_Count  = len(self.word_Index_Dict)
        
        orthography_Pattern = np.zeros((pattern_Count, self.orthography_Size), dtype= np.int32)
        phonology_Pattern = np.zeros((pattern_Count, self.max_Pronunciation_Length, self.phonology_Size), dtype= np.float32)
        frequency_Pattern = np.zeros((pattern_Count), dtype= np.float32)

        for word, index in self.word_Index_Dict.items():
            orthography_Pattern[index] = self.Word_to_Pattern(word)
            phonology_Pattern[index] = self.Pronunciation_to_Pattern(self.pronunciation_Dict[word])
            frequency_Pattern[index] = self.frequency_Dict[word]
            
        #Queue
        for epoch in range(self.start_Epoch, self.max_Epoch):
            pattern_Index_List = np.arange(pattern_Count)
            if hp_Dict['Use_Frequency']:
                pattern_Index_List = pattern_Index_List[np.random.rand(pattern_Count) < frequency_Pattern]
            self.trained_Pattern_Index_Dict[epoch] = pattern_Index_List
            shuffle(pattern_Index_List)
            pattern_Index_Batch_List = [pattern_Index_List[x:x+hp_Dict['Batch_Size']] for x in range(0, len(pattern_Index_List), hp_Dict['Batch_Size'])]
            
            current_Index = 0
            is_New_Epoch = True
            while current_Index < len(pattern_Index_Batch_List):
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Pattern_Queue']:
                    time.sleep(0.1)
                    continue
                             
                selected_Orthography_Pattern = orthography_Pattern[pattern_Index_Batch_List[current_Index]]
                selected_Phonology_Pattern = phonology_Pattern[pattern_Index_Batch_List[current_Index]]
                
                self.pattern_Queue.append([
                    epoch,
                    is_New_Epoch,
                    selected_Orthography_Pattern,
                    selected_Phonology_Pattern
                    ])
                
                current_Index += 1
                is_New_Epoch = False

        self.is_Finished = True

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.01)
        return self.pattern_Queue.popleft()

    def Get_Inference_Pattern_Bak(self, letter_String_List, added_Pronunciation_Dict = {}):
        orthography_Pattern = np.vstack([self.Word_to_Pattern(letter_String) for letter_String in letter_String_List]).astype(np.int32)

        pattern_Index_List = list(range(len(letter_String_List)))
        pattern_Index_Batch_List = [pattern_Index_List[x:x+hp_Dict['Batch_Size']] for x in range(0, len(pattern_Index_List), hp_Dict['Batch_Size'])]        
        split_Orthography_Pattern_List = [orthography_Pattern[pattern_Index_Batch] for pattern_Index_Batch in pattern_Index_Batch_List]

        word_Label_Indices = np.array([
            self.word_Index_Dict[letter_String] if letter_String in self.word_Index_Dict.keys() else -1
            for letter_String in letter_String_List
            ], dtype= np.int32)

        phoneme_Index_Dict = {phoneme: index for index, phoneme in enumerate(self.phoneme_List)}
        phoneme_Label_Indices_List = []
        for letter_String in letter_String_List:
            if letter_String in added_Pronunciation_Dict.keys():
                pronunciation = added_Pronunciation_Dict[letter_String]
            elif letter_String in self.pronunciation_Dict.keys():
                pronunciation = self.pronunciation_Dict[letter_String]
            else:
                raise ValueError('There is no pronunciation information of {}.'.format(letter_String))
            pronunciation = pronunciation + '_' * (self.max_Pronunciation_Length - len(pronunciation))
            phoneme_Label_Indices_List.append(np.array([phoneme_Index_Dict[phoneme] for phoneme in pronunciation], dtype=np.int32))
        

        return split_Orthography_Pattern_List, word_Label_Indices, np.array(phoneme_Label_Indices_List, dtype=np.int32)

    def Get_Inference_Pattern(self, letter_String_List, added_Pronunciation_Dict = {}):
        '''
        added_Pronunciation_Dict: key is letter string, value is 'str' or sequence of 'str'.
        '''
        index_Added_Pronunciation_Dict = {}
        for pronunciation in added_Pronunciation_Dict.values():
            if isinstance(pronunciation, str):
                if not pronunciation in index_Added_Pronunciation_Dict.keys():
                    index_Added_Pronunciation_Dict[len(index_Added_Pronunciation_Dict)] = pronunciation
            elif isinstance(pronunciation, Sequence):
                for heteronym in pronunciation:
                    if not heteronym in index_Added_Pronunciation_Dict.values():
                        index_Added_Pronunciation_Dict[len(index_Added_Pronunciation_Dict)] = heteronym
            else:
                raise ValueError('the value of \'added_Pronunciation_Dict\' must be str or Sequence: The inserted type: {}'.format(type(pronunciation)))

        added_Word_Labels = np.zeros(
            shape= (len(index_Added_Pronunciation_Dict), self.max_Pronunciation_Length, self.phonology_Size),
            dtype= np.float32
            )
        for index, pronunciation in index_Added_Pronunciation_Dict.items():
            added_Word_Labels[index] = self.Pronunciation_to_Pattern(pronunciation)

        added_Pronunciation_Index_Dict = {
            pronunciation: index + self.word_Labels.shape[0]
            for index, pronunciation in index_Added_Pronunciation_Dict.items()
            }

        inference_Tuple_List = []
        for letter_String in letter_String_List:
            if letter_String in added_Pronunciation_Dict.keys():                
                pronunciation = added_Pronunciation_Dict[letter_String]
                if isinstance(pronunciation, str):
                    inference_Tuple_List.append((
                        letter_String,
                        pronunciation,
                        added_Pronunciation_Index_Dict[pronunciation]
                        ))
                elif isinstance(pronunciation, Sequence):
                    for heteronym in pronunciation:
                        inference_Tuple_List.append((
                        letter_String,
                        heteronym,
                        added_Pronunciation_Index_Dict[heteronym]
                        ))
            elif letter_String in self.word_Index_Dict.keys():
                inference_Tuple_List.append((
                    letter_String,
                    self.pronunciation_Dict[letter_String],
                    self.word_Index_Dict[letter_String]
                    ))
            else:
                raise ValueError('There is no pronunciation information of {}.'.format(letter_String))

        letter_String_List, pronunciation_List, word_Label_Index_List = zip(*inference_Tuple_List)

        orthography_Pattern = np.vstack([self.Word_to_Pattern(letter_String) for letter_String in letter_String_List]).astype(np.int32)
        pattern_Index_List = list(range(len(letter_String_List)))
        pattern_Index_Batch_List = [pattern_Index_List[x:x+hp_Dict['Batch_Size']] for x in range(0, len(pattern_Index_List), hp_Dict['Batch_Size'])]        
        split_Orthography_Pattern_List = [orthography_Pattern[pattern_Index_Batch] for pattern_Index_Batch in pattern_Index_Batch_List]

        word_Label_Indices = np.array(word_Label_Index_List, dtype= np.int32)        

        phoneme_Index_Dict = {phoneme: index for index, phoneme in enumerate(self.phoneme_List)}
        phoneme_Label_Indices_List = []
        for pronunciation in pronunciation_List:
            pronunciation = pronunciation + '_' * (self.max_Pronunciation_Length - len(pronunciation))
            phoneme_Label_Indices_List.append(np.array([phoneme_Index_Dict[phoneme] for phoneme in pronunciation], dtype=np.int32))
        phoneme_Label_Indices = np.array(phoneme_Label_Indices_List, dtype=np.int32)

        return letter_String_List, pronunciation_List, split_Orthography_Pattern_List, word_Label_Indices, phoneme_Label_Indices, added_Word_Labels

if __name__ == '__main__':
    new_Feeder = Feeder(
        start_Epoch = 0, 
        max_Epoch = 1000, 
        )
        
    print(new_Feeder.Get_Inference_Pattern('aaa'))