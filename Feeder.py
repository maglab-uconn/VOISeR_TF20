import numpy as np;
import tensorflow as tf;
from threading import Thread;
from collections import deque, Sequence;
from random import shuffle;
import time, json;

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

def Load_Data():    # Load lexicon. This function is used both of 'Feeder' and 'Model'.
    data_Dict = {}

    with open (hp_Dict['Train']['Pattern_File_Path'], 'r') as f:    #Load lexicon
        readLines = f.readlines()[1:]

    # lexicon information to dict objects
    splited_ReadLine = [readLine.strip().split(',')[1:] for readLine in readLines]
    data_Dict['Word_Index_Dict'] = {word.lower(): index for index, (word, _, _, _, _) in enumerate(splited_ReadLine)}
    data_Dict['Pronunciation_Dict'] = {word.lower(): pronunciation for word, pronunciation, _, _, _ in splited_ReadLine}
    data_Dict['Frequency_Dict'] = {word.lower(): float(frequency) * 0.05 + 0.1 for word, _, _, frequency, _ in splited_ReadLine}
    data_Dict['Human_RT_Dict'] = {word.lower(): float(rt) for word, _, _, _, rt in splited_ReadLine}
    
    data_Dict['Max_Word_Length'] = max([len(word) for word in data_Dict['Word_Index_Dict'].keys()])
    data_Dict['Max_Pronunciation_Length'] = max([len(pronunciation) for pronunciation in data_Dict['Pronunciation_Dict'].values()])

    #Currently, one word must have one pronunciation. If you want to train `heteronym`, please remove these two liens
    if len(data_Dict['Word_Index_Dict'].keys()) != len(set(data_Dict['Word_Index_Dict'].keys())):
        raise Exception('More than one word of the same spelling is in the data!')

    #Letter and phoneme information
    letter_Set = set()
    phoneme_Set = set()
    for word, pronunciation in data_Dict['Pronunciation_Dict'].items():
        letter_Set.update(set(word))    # A set cannot store duplicate data. Thus, only one letter is stored.
        phoneme_Set.update(set(pronunciation))  # A set cannot store duplicate data. Thus, only one phoneme is stored.
    data_Dict['Letter_List'] = list(sorted(list(letter_Set))) + ['_']   # Adding silence
    data_Dict['Phoneme_List'] = list(sorted(list(phoneme_Set))) + ['_'] # Adding silence
        
    #There are two key types: [letter], [slot_Index, letter]
    #Value is index of each letter.
    data_Dict['Letter_Index_Dict'] = {letter: letter_Index for letter_Index, letter in enumerate(data_Dict['Letter_List'])}
    for slot_Index in range(data_Dict['Max_Word_Length']):
        for letter_Index, letter in enumerate(data_Dict['Letter_List']):
            data_Dict['Letter_Index_Dict'][slot_Index, letter] = slot_Index * len(data_Dict['Letter_List']) + letter_Index

    #There are two key types: [phoneme], [slot_Index, phoneme]
    #Value is index of each phoneme.
    data_Dict['Phoneme_Index_Dict'] = {phoneme: phoneme_Index for phoneme_Index, phoneme in enumerate(data_Dict['Phoneme_List'])}
    for slot_Index in range(data_Dict['Max_Pronunciation_Length']):
        for phoneme_Index, phoneme in enumerate(data_Dict['Phoneme_List']):
            data_Dict['Phoneme_Index_Dict'][slot_Index, phoneme] = slot_Index * len(data_Dict['Phoneme_List']) + phoneme_Index

    if not hp_Dict['Phoneme_Feature_File_Path'] is None:    # Feature based target phoneme vector load
        with open (hp_Dict['Phoneme_Feature_File_Path'], 'r') as f:
            readLines = f.readlines()[1:]
        splited_ReadLine = [readLine.strip().split(',') for readLine in readLines]
        data_Dict['Phoneme_Pattern_Dict'] = {pattern[0]: np.array([float(x) for x in pattern[1:]], dtype=np.float32) for pattern in splited_ReadLine}
    else:   # One-hot based target phoneme vector load
        data_Dict['Phoneme_Pattern_Dict'] = {}
        for phoneme in data_Dict['Phoneme_List']:
            new_Pattern = np.zeros(shape = len(data_Dict['Phoneme_List']))  #All units are 1
            new_Pattern[data_Dict['Phoneme_Index_Dict'][phoneme]] = 1   #Only one unit is 1
            data_Dict['Phoneme_Pattern_Dict'][phoneme] = new_Pattern
                    
    data_Dict['Orthography_Size'] = data_Dict['Max_Word_Length']    #Orthograhy layer size is maximum word length
    data_Dict['Phonology_Size'] = data_Dict['Phoneme_Pattern_Dict'][data_Dict['Phoneme_List'][0]].shape[0]  #Phonlogy layer size is one phoneme vector length. It is different by feature based or one-hot based. This is one step's size.

    return data_Dict

class Feeder:
    def __init__(self, start_Epoch, max_Epoch):
        self.start_Epoch = start_Epoch
        self.max_Epoch = max_Epoch
        
        self.Load_Data()
        self.Analyzer_Label_Generate()  #Label of each

        self.is_Finished = False    #Checking training done. When all training pattern generated until last epoch, this value is changed to True.
        self.pattern_Queue = deque()    #Storage space for training pattern
        self.trained_Pattern_Index_Dict = {}    #Saving the trained count of each patterh
                
        #By threading, pattern is generated parallely
        pattern_Generate_Thread = Thread(target=self.Pattern_Generate)
        pattern_Generate_Thread.daemon = True
        pattern_Generate_Thread.start()
    
    def Load_Data(self):    #This function just use 'Load_Data' function. Please see 'Load_Data' function.
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
        
    def Analyzer_Label_Generate(self):  #This function is for the inference generate. Clearer, this data is used at result analyzing.
        index_Word_Dict = {index: word for word, index in self.word_Index_Dict.items()} #Matching index to word 

        self.word_Labels = np.stack([
            self.Pronunciation_to_Pattern(self.pronunciation_Dict[index_Word_Dict[index]])
            for index in range(len(index_Word_Dict))
            ])  # Generate a single numpy array that all words' pronunciation patterns are stacked.

        self.phoneme_Labels = np.vstack([self.phoneme_Pattern_Dict[phoneme] for phoneme in self.phoneme_List])  # Generate a single numpy array that all phoneme patterns are stacked.
    
    def Word_to_Pattern(self, word):
        word = word + '_' * (self.max_Word_Length - len(word))  #Padding is filled by letter '_'.

        return np.array([self.letter_Index_Dict[letter] for letter in word], dtype=np.int32)    #word to vector

    def Pronunciation_to_Pattern(self, pronunciation):
        pronunciation = pronunciation + '_' * (self.max_Pronunciation_Length - len(pronunciation))  #Padding is filled by letter '_'.

        new_Pattern = np.zeros((self.max_Pronunciation_Length, self.phonology_Size))    #pronunciation to matrix, shape: [Step, Phoneme vector size]
        for index, phoneme in enumerate(pronunciation):
            new_Pattern[index] = self.phoneme_Pattern_Dict[phoneme] #Each step zero vector is replaced to phoneme vector
                    
        return new_Pattern

    def Pattern_Generate(self): #Training pattern generate.
        #Batched Pattern Making
        pattern_Count  = len(self.word_Index_Dict)  # Training pattern count
        
        orthography_Pattern = np.zeros((pattern_Count, self.orthography_Size), dtype= np.int32) #Numpy array of input pattern,  [Patterns, orthography size]
        phonology_Pattern = np.zeros((pattern_Count, self.max_Pronunciation_Length, self.phonology_Size), dtype= np.float32)    #Numpy array of target pattern,  [Patterns, step, phonology size]
        frequency_Pattern = np.zeros((pattern_Count), dtype= np.float32)    #Numpy array of pattern training probability,  [Patterns]

        for word, index in self.word_Index_Dict.items():    #Each zero pattern is replaced to real pattern
            orthography_Pattern[index] = self.Word_to_Pattern(word)
            phonology_Pattern[index] = self.Pronunciation_to_Pattern(self.pronunciation_Dict[word])
            frequency_Pattern[index] = self.frequency_Dict[word]
            
        #Queue
        for epoch in range(self.start_Epoch, self.max_Epoch):   #From start epoch to end epoch
            pattern_Index_List = np.arange(pattern_Count)   #Generating all pattern indices
            if hp_Dict['Use_Frequency']:    #If use frequency, only patterns that pass the probability test are used for learning at each epoch
                pattern_Index_List = pattern_Index_List[np.random.rand(pattern_Count) < frequency_Pattern]
            self.trained_Pattern_Index_Dict[epoch] = pattern_Index_List #Saving the trained index list
            shuffle(pattern_Index_List) #Shuffle for randomizing
            pattern_Index_Batch_List = [pattern_Index_List[x:x+hp_Dict['Batch_Size']] for x in range(0, len(pattern_Index_List), hp_Dict['Batch_Size'])]    #Split index list to genrate batchs
            
            current_Index = 0   #Batch index
            is_New_Epoch = True #Checking whether first training of each epoch for test for checkpoint save
            while current_Index < len(pattern_Index_Batch_List):
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Pattern_Queue']:    #If queue is full, pattern generating is stopped while 0.1 sec.
                    time.sleep(0.1)
                    continue
                             
                selected_Orthography_Pattern = orthography_Pattern[pattern_Index_Batch_List[current_Index]] #Getting batch input patterns
                selected_Phonology_Pattern = phonology_Pattern[pattern_Index_Batch_List[current_Index]] #Getting batch target patterns
                
                self.pattern_Queue.append([
                    epoch,
                    is_New_Epoch,
                    selected_Orthography_Pattern,
                    selected_Phonology_Pattern
                    ])  #Storing generated pattern.
                
                current_Index += 1  #Batch index + 1
                is_New_Epoch = False    #Next pattern batch is not new epoch.

        self.is_Finished = True # Training pattern generating done.

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #If queue is empty, waiting 0.01 sec.
            time.sleep(0.01)
        return self.pattern_Queue.popleft() #Return by FIFO rule.

    def Get_Inference_Pattern(self, letter_String_List, added_Pronunciation_Dict = {}):
        '''
        added_Pronunciation_Dict: key is letter string, value is 'str' or sequence of 'str'.
        '''
        # nonword pronunciations are added to pronuniciation dict for analyizer
        index_Added_Pronunciation_Dict = {} 
        for pronunciation in added_Pronunciation_Dict.values():
            if isinstance(pronunciation, str):  #If nonword has only one pronunciation.
                if not pronunciation in index_Added_Pronunciation_Dict.keys():  #If there is already same pronunciation in index_Added_Pronunciation_Dict, passed.
                    index_Added_Pronunciation_Dict[len(index_Added_Pronunciation_Dict)] = pronunciation
            elif isinstance(pronunciation, Sequence):   #If nonword has multiple pronunciations.
                for heteronym in pronunciation:
                    if not heteronym in index_Added_Pronunciation_Dict.values():    #If there is already same pronunciation in index_Added_Pronunciation_Dict, passed.
                        index_Added_Pronunciation_Dict[len(index_Added_Pronunciation_Dict)] = heteronym
            else:
                raise ValueError('the value of \'added_Pronunciation_Dict\' must be str or Sequence: The inserted type: {}'.format(type(pronunciation)))
        
        added_Word_Labels = np.zeros(   # Nonword pronunciation pattern are added to word labels for analyizer
            shape= (len(index_Added_Pronunciation_Dict), self.max_Pronunciation_Length, self.phonology_Size),
            dtype= np.float32
            )
        for index, pronunciation in index_Added_Pronunciation_Dict.items():
            added_Word_Labels[index] = self.Pronunciation_to_Pattern(pronunciation)

        added_Pronunciation_Index_Dict = {  # Nonword pronunciations also need specific indices.
            pronunciation: index + self.word_Labels.shape[0]
            for index, pronunciation in index_Added_Pronunciation_Dict.items()
            }
                
        inference_Tuple_List = []   #Generating pattern by letter string list, trained pronunciation dict and added pronunciation dict.
        for letter_String in letter_String_List:
            #If the letter string is trained word, model already has the letter string's correct pronunciation. However, if there is another pronunciation in added_Pronunciation_Dict, the added_Pronunciation_Dict's pronunciation is used preferentially.
            if letter_String in added_Pronunciation_Dict.keys():    #If there is a pronunciation in added pronunciation dict.
                pronunciation = added_Pronunciation_Dict[letter_String]
                if isinstance(pronunciation, str):  #If nonword has only one pronunciation, one letter string generate one pattern.
                    inference_Tuple_List.append((
                        letter_String,
                        pronunciation,
                        added_Pronunciation_Index_Dict[pronunciation]
                        ))
                elif isinstance(pronunciation, Sequence):   #If nonword has multiple pronunciations, one letter string generate several patterns.
                    for heteronym in pronunciation:
                        inference_Tuple_List.append((
                        letter_String,
                        heteronym,
                        added_Pronunciation_Index_Dict[heteronym]
                        ))
            elif letter_String in self.word_Index_Dict.keys():  #If letter string is trained word.
                inference_Tuple_List.append((
                    letter_String,
                    self.pronunciation_Dict[letter_String],
                    self.word_Index_Dict[letter_String]
                    ))
            else:
                raise ValueError('There is no pronunciation information of {}.'.format(letter_String))

        letter_String_List, pronunciation_List, word_Label_Index_List = zip(*inference_Tuple_List)

        orthography_Pattern = np.vstack([self.Word_to_Pattern(letter_String) for letter_String in letter_String_List]).astype(np.int32) #stacked input pattern
        pattern_Index_List = list(range(len(letter_String_List)))   #patterns' indices
        pattern_Index_Batch_List = [pattern_Index_List[x:x+hp_Dict['Batch_Size']] for x in range(0, len(pattern_Index_List), hp_Dict['Batch_Size'])]    #Splitted pattern index by batch
        split_Orthography_Pattern_List = [orthography_Pattern[pattern_Index_Batch] for pattern_Index_Batch in pattern_Index_Batch_List] #splitted input pattern by batch

        word_Label_Indices = np.array(word_Label_Index_List, dtype= np.int32)   #For analyzing, generating a numpy array about all patterns' indices

        #For analyzing, generating a numpy array about the indices of all phonemes of pronunciation of patterns
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