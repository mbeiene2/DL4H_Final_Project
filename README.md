# DL4H_Final_Project - Reproducing research paper - Towards automated clinical coding

This repository is the official implementation of DL4H_Final_Project - Reproducing research paper - Towards automated clinical coding. This project is part of class project for Deep Learning for Healthcare at UIUC 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training, and Evaluation

There are four models implemented for this project. They models are implemented for the 4 levels of ontology for ICD-9-CM code. Each level data preparation, training, and evaluation are in one python file. To perform only one operation but comment out other section of the code. I have chosen to do this to make the management of the code much easier

Level_1_complete.py – Data preparation, training, and evaluation of level 1 ontology for labels of discharge summary using GRU(X)-GRU(Z) model and tfidf(d) atomic model

To train and evaluate level 1 models (GRU(X)-GRU(Z) and tfidf(d) atomic model, run this command:

```level 1
python Level_1_complete.py 
```
Level_2_complete.py – Data preparation, training, and evaluation of level 2 ontology for labels of discharge summary using GRU(X)-GRU(Z) model and tfidf(d) atomic model

To train and evaluate level 2 models (GRU(X)-GRU(Z) and tfidf(d) atomic model, run this command:

```level 2
python Level_2_complete.py 
```
Level_3_complete.py – Data preparation, training, and evaluation of level 3 ontology for labels of discharge summary using GRU(X)-GRU(Z) model and tfidf(d) atomic model

Level_3.pkl – Input data for level 3 labels (ICD9-CM) code. Since the number of values is larger than level 1 and level 2, this list is not included with the code

To train and evaluate level 3 models (GRU(X)-GRU(Z) and tfidf(d) atomic model, run this command:

```level 3
python Level_3_complete.py 
```
Level_4_complete.py – Data preparation, training, and evaluation of level 4 ontology for labels of discharge summary using GRU(X)-GRU(Z) model and tfidf(d) atomic model

Level_4.pkl – Input data for level 4 labels (ICD9-CM) code. Since the number of values is larger than level 1 and level 2, this list is not included with the code

To train and evaluate level 4 models (GRU(X)-GRU(Z) and tfidf(d) atomic model, run this command:

```level 4
python Level_4_complete.py 
```

## Results

For detail explanation and result, please see the published paper. 

Here is high level result

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

Level 1 - Level_1_complete.py 

| Model name         | F1 Original     | F1 Reproduced  |
| ------------------ |---------------- | -------------- |
| GRU(X)-GRU(Z)      |     0.271       |      0.263     |
| tfidf(d)-atomic    |     0.262       |      0.254     |

Level 2 - Level_2_complete.py 

| Model name         | F1 Original     | F1 Reproduced  |
| ------------------ |---------------- | -------------- |
| GRU(X)-GRU(Z)      |     0.544       |      0.530     |
| tfidf(d)-atomic    |     0.537       |      0.520     |

Level 3 - Level_3_complete.py 

| Model name         | F1 Original     | F1 Reproduced  |
| ------------------ |---------------- | -------------- |
| GRU(X)-GRU(Z)      |     0.419       |      0.413     |
| tfidf(d)-atomic    |     0.409       |      0.400     |

Level 4 - Level_4_complete.py 

| Model name         | F1 Original     | F1 Reproduced  |
| ------------------ |---------------- | -------------- |
| GRU(X)-GRU(Z)      |     0.688       |      0.673     |
| tfidf(d)-atomic    |     0.672       |      0.659     |


## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at mbeiene@illinois.edu or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
