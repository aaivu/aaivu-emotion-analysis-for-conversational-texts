
# Emotion-Analysis-for-Conversational-Texts

![project] ![research]



- <b>Project Mentor</b>
    1. Dr Uthayasanker Thayasivam
- <b>Contributors</b>
    1. Piruntha Navanesan
    2. Jarsigan Vickneswaran
    3. Vahesan Vijayaratnam


---

## Overview

We are developing a more effective and efficient system that would recognize the emotions of conversational texts. Effectiveness of the system would be addressing the high accuracy of the results while the efficiency would ideally provide the best results using a comparatively small data set.


## Dataset
We were able to obtain a well-structured data set from Microsoft through the “EmoContext” competition. Data collection process done by the competition organizers is explained below.
<br> Source of Dataset : 
<br> https://competitions.codalab.org/competitions/19790#learn_the_details-data-set-format


## Source Code of the Model

- The models were trained using Keras with TensorFlow backend.

## Requirements

- Tensorflow
- Keras
- nltk
- Python 3.5 or above
- OS: Ubuntu

## Pre-Trained Embeddings

We are using customized FastText embedding as the pre-trained embedding model Which is a context-free word embedding trained with 322M tweets that are mostly emotion related. It generates better word embeddings for rare words, or even words not seen during training because it uses n-gram characters.

## How to run

1- Install all necessary requirements.      
2- Download source code from github and add them into a folder.     
3- Download a pre-trained word embedding and add into the same folder.   
4- Specify the embedding file name in the baseline file.    
5- Run the following command to run the model.   
    python baseline_with_eval_With_Nltk.py -config testBaseline.config

## Files Description


|  File Name	|  Description	| 
|:---------|:-----------	|
| baseline_with_eval_With_Nltk.py    	| Contains the code basics for the model  	|
| testBaseline.config     	| Contains main parameters     	| 
|Train.txt   	| Contains Training data     	|
|Devwithoutlabels.txt   	| Contains test data      	|
|SolFile.txt   	| Contains the result data     	|


## Results



| Emotion 	| Precision 	| Recall 	| Micro F1 	|
|:---------:|:-----------:	|:---------:|:----------:|
| Happy   	| 0.696     	| 0.750  	| 0.722    	|
| Sad     	| 0.472     	| 0.760  	| 0.751    	|
| Angry   	| 0.716     	| 0.795  	| 0.754    	|


## Limitations in our final model

- Emoji prediction is weak in our model.
- Overfitting for some emotional related words.
- Censored words are not handled.

## Achievements

- Achieved a best micro F1 value (0.7420) which betters the 3rd Quartile value of 0.7317 and stands up into the top quarter of the leaderboard of EmoContext competition.
- When ranking the models in terms of recall for happy emotion, our model outperforms all other models.
- Providing the simplest and easily referable emotion prediction model for future researchers.


## More references

1. Reference
2. Link

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen

