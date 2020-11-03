# UPST
A semi-training sentence classification framework based on user polarity

## requirements

```
python == 3.6
numpy
pytorch
allennlp
```

## training in train_data and testing in dev_data
```python self_training_question_classification.py -iter 10 -batch 128 -SC_u 1 -polarity 1 -unlabel 1```


