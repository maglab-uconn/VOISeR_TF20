import numpy as np;
import tensorflow as tf;
from threading import Thread;
from collections import deque;
from random import shuffle;
import time, json;

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

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
        with open (hp_Dict['Train']['Pattern_File_Path'], 'r') as f:
            readLines = f.readlines()[1:]

        splited_ReadLine = [readLine.strip().split(',')[1:] for readLine in readLines]
        self.word_Index_Dict = {word.lower(): index for index, (word, _, _, _, _) in enumerate(splited_ReadLine)}
        self.pronunciation_Dict = {word.lower(): pronunciation for word, pronunciation, _, _, _ in splited_ReadLine}
        self.frequency_Dict = {word.lower(): float(frequency) * 0.05 + 0.1 for word, _, _, frequency, _ in splited_ReadLine}
        self.human_RT_Dict = {word.lower(): float(rt) for word, _, _, _, rt in splited_ReadLine}
        
        self.max_Word_Length = max([len(word) for word in self.word_Index_Dict.keys()])
        self.max_Pronunciation_Length = max([len(pronunciation) for pronunciation in self.pronunciation_Dict.values()])

        if len(self.word_Index_Dict.keys()) != len(set(self.word_Index_Dict.keys())):
            raise Exception('More than one word of the same spelling is in the data!')

        letter_Set = set()
        phoneme_Set = set()
        for word, pronunciation in self.pronunciation_Dict.items():
            letter_Set.update(set(word))
            phoneme_Set.update(set(pronunciation))
        self.letter_List = list(sorted(list(letter_Set))) + ['_']
        self.phoneme_List = list(sorted(list(phoneme_Set))) + ['_']
            
        #There are two key types: [letter], [slot_Index, letter]
        self.letter_Index_Dict = {letter: letter_Index for letter_Index, letter in enumerate(self.letter_List)}
        for slot_Index in range(self.max_Word_Length):
            for letter_Index, letter in enumerate(self.letter_List):
                self.letter_Index_Dict[slot_Index, letter] = slot_Index * len(self.letter_List) + letter_Index

        #There are two key types: [phoneme], [slot_Index, phoneme]
        self.phoneme_Index_Dict = {phoneme: phoneme_Index for phoneme_Index, phoneme in enumerate(self.phoneme_List)}
        for slot_Index in range(self.max_Pronunciation_Length):
            for phoneme_Index, phoneme in enumerate(self.phoneme_List):
                self.phoneme_Index_Dict[slot_Index, phoneme] = slot_Index * len(self.phoneme_List) + phoneme_Index

        if not hp_Dict['Phoneme_Feature_File_Path'] is None:
            with open (hp_Dict['Phoneme_Feature_File_Path'], 'r') as f:
                readLines = f.readlines()[1:]
            splited_ReadLine = [readLine.strip().split(',') for readLine in readLines]
            self.phoneme_Pattern_Dict = {pattern[0]: np.array([float(x) for x in pattern[1:]], dtype=np.float32) for pattern in splited_ReadLine}
        else:
            self.phoneme_Pattern_Dict = {}
            for phoneme in self.phoneme_List:
                new_Pattern = np.zeros(shape = len(self.phoneme_List))
                new_Pattern[self.phoneme_Index_Dict[phoneme]] = 1
                self.phoneme_Pattern_Dict[phoneme] = new_Pattern
                        
        self.orthography_Size = self.max_Word_Length
        self.phonology_Size = self.phoneme_Pattern_Dict[self.phoneme_List[0]].shape[0]
        
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

    def Get_Inference_Pattern(self, letter_String_List, added_Pronunciation_Dict = {}):
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


if __name__ == '__main__':
    new_Feeder = Feeder(
        start_Epoch = 0, 
        max_Epoch = 1000, 
        )
        
    print(new_Feeder.Get_Inference_Pattern())