# VOISeR model for TF20

VOISeR is a simple computational reading model to support the friends-and-enemies research.  

## Differences from Earlier version

* TensorFlow 2.0 compatiblity
* Some arg parser parameters are moved to 'Hyper_Paremeter.json' file.
* Analyzer is in the main code.
* Continue learning function from saved checkpoint added

## Requirement

tensorflow >= 2.0

## Structure
![Structure](https://user-images.githubusercontent.com/17133841/66222115-70035a80-e69e-11e9-8a8e-0bc0ef4c71d7.png)

* The using of 'Orthography → Hidden' and 'Hidden → Hidden' is selectable.
* The model reported in the paper has both connections (O->H and H->H).

## Dataset

Data were obtained from 'The English Lexicon Project':

    Balota, D. A., Yap, M. J., Hutchison, K. A., Cortese, M. J., Kessler, B., Loftis, B., ... & Treiman, R. (2007). The English lexicon project. Behavior research methods, 39(3), 445-459.
    
The "ELP_groupData.csv" file was used to train VOISeR.

## Hyper parameters

* Use_Frequency
    * If this parameter is true, model will use the frequency information of words in the training.

* Orthography_Embedding_Size
    * If this parameter is integer, model use the embedding about the orthographic input.
    * The inserted integer value become the size of embedding.
    * If 'null', orthography is one hot structure.

* Phoneme_Feature_File_Path
    * If this parameter is a file path, the target pattern become the distributed pattern.
    * If 'null', the target pattern become one-hot structure.
    * Please see the example: 'phonetic_feature_definitions_18_features.csv'

* RNN
    * Size
        * Determines the size of the hidden layer
        * A positive integer is required.
    * Use_Feedback
        * Determines output layer's previous activation is used for the hidden activation calculation.
    * Use_Recurrent
        * Determines hidden layer's previous activation is used for the hidden activation calculation.

* Train
    * Pattern_File_Path
        * File path which has the word information to be used for learning
        * Please see the example: 'ELP_groupData.csv'
    * Max_Pattern_Queue
        * Determines the maximum size of queue saving the next training pattern
    * Learning_Rate
        * Determine the size of learning rate.
        * A positive float is required.
    * Inference_Timing
        * Determine the frequency of the inference during learning.
        * A positive integer is required.
    * Checkpoint_Save_Timing
        * Determine the frequency of the checkpoint saving during learning
        * A positive integer is required.

* Analyzer
    * Batch_Size
        * Determine the batch size during analysis.
        * When Out of memory occurs, decrease according to the environment(GPU memory).

* Batch_Size
    * Determine the batch size during learning.
    * When Out of memory occurs, decrease according to the environment(GPU memory).

* Export_Path
    * Inference result and checkpoint save path

## Run

### Command
    python Model.py [parameters]
    
### Parameters

* `-e <int>`
    * Determine the model's maximum training epoch.
    * This parameter is required.

* `-se <int>`
    * Determine the model's start training epoch.
    * When this parameter set, model require the checkpoint of corresponding epcoh.
    
* `-idx <int>`
    * Attach an tag to each result directory.
    * This value does not affect the performance of the model.

## Inference

### Method

1. In terminal, type 'ipython'

2. Type the following commands with your 'using_Epoch' and 'export_Path':
```
from Model import VOISeR

using_Epoch= <int>
export_Path= <path>

new_VOISeR = VOISeR(
    start_Epoch= using_Epoch,
    max_Epoch= using_Epoch,
    export_Path= export_Path
    )
new_VOISeR.restore(start_Epoch= using_Epoch)
```

3. Set 'letter_String_List' and 'added_Pronunciation_Dict'.
    * If letter strings are trained words:
        ```
        letter_String_List = <word list>
        added_Pronunciation_Dict = {}
        ```
    * If letter strings are non-trained words(nonwords):
        ```
        letter_String_List = [<str>]
        added_Pronunciation_Dict = {<str>: <pronunciation>}        
        ```
    * The pronunciations of all nonword must be set at add_Pronunciation_Dict

4. Type the following commands:
```
new_VOISeR.Inference(
    letter_String_List = letter_String_List
    added_Pronunciation_Dict = added_Pronunciation_Dict
    )
```

5. Please check 'Inference.Summary.txt' in the inference directory. 