# SemEval 2024 Task 1 (Team: UMBCLU)

## Method Name
TranSem

### Description
In this model, I first translated the data using 4 different models and use that data to train a different model.

Later during inference, that test data is also translated using one of the models and then trained model is used to predict the output.


### Best Hyperparameters
```
 'accumulate_grad_batches': 32,
 'batch_size': 16,
 'cuda': True,
 'data_dir': './data/Track A',
 'ddp': False,
 'debug': False,
 'early_stopping_patience': 10,
 'enc_dropout': 0.1,
 'enc_pooling': 'mean',
 'lr': 1e-05,
 'max_epochs': -1,
 'model_name': 'sentence-transformers/all-distilroberta-v1',
 'monitoring_metric': 'valid/corr',
 'monitoring_mode': 'max',
 'seed': 42,
 'validate_every': 1.0,
 'weight_decay': 0.01
```

### Test Results
| Language | Spearman Correlation |
|----------|----------------------|
| English  | 0.8124657642704284   |
| Hausa    | 0.6402557259721893   |
| Kinyarwanda | 0.6806672639024565 |
| Marathi  | 0.8406501120112206   |
| Moroccan Arabic | 0.7447707874574931 |
| Spanish  | 0.6382694075818184   |
| Telugu   | 0.8255452084837941   |

## Team Name
UMBCLU

### Team Members
1. Shubhashis Roy Dipta
2. Sai Vallurupalli 

#### Affiliation
University of Maryland, Baltimore County