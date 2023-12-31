# Fine-tuning - IFRE
Inplementation of "Relative Fine-tuning in Incremental Few-Shot Relation Extraction"

### Environments
- ``python 3``
- ``PyTorch 1.7.1``
- ``transformers 4.6.0``

### Datasets and Models
You can find the training and validation data here: [FewRel 1.0 data](https://github.com/thunlp/FewRel/tree/master/data). For the test data, you can easily download from FewRel 1.0 competition website: https://competitions.codalab.org/competitions/27980

We release our trained models using BERT and CP as backend models respectively at [Google Drive](https://drive.google.com/drive/folders/1_mIg5QfIl2FuSDVn3_n7SNV9AfZNw4tL?usp=sharing). The file structure as below:

```
--BERT
    --nodropPrototype-nodropRelation-lr-1e-5
--CP
    --nodropPrototype-nodropRelation-lr-9e-6
    --nodropPrototype-nodropRelation-lr-5e-6
```
You can reproduce our result in the paper with models in *BERT/nodropPrototype-nodropRelation-lr-1e-5* and *CP/nodropPrototype-nodropRelation-lr-5e-6*. We also provide the trained model with a different learning rate for CP in *CP/--nodropPrototype-nodropRelation-lr-9e-6* for extra reference.


### Code
Put all data in the **data** folder, CP pretrained model in the **CP_model** folder (you can download CP model from https://github.com/thunlp/RE-Context-or-Names/tree/master/pretrain or [Google Drive](https://drive.google.com/drive/folders/1AwQLqlHJHPuB1aKJ8XPHu8nu237kgtWj?usp=sharing)), and then you can simply use three scripts: *run_train.sh*, *run_eval.sh*, *run_submit.sh* for train, evaluation and test.

#### Train
Set the corresponding parameter values in the script, and then run:
```
sh run_train.sh
```
Some explanations of the parameters in the script:
```
--pretrain_ckpt
	the path for the BERT-base-uncased
--backend_model
	bert or cp, select one backend model
```
#### Evaluation
Set the corresponding parameter values in the script, and then run:
```
sh run_eval.sh
```
Some explanations of the parameters in the script:
```
--test_iter
	1000, the evaluation iteration
--load_ckpt
	the path of the trained model
```
#### Test
Set the corresponding parameter values in the script, and then run:
```
sh run_submit.sh
```
Some explanations of the parameters in the script:
```
--test_output
	the path to save the prediction file
```

### Results

**BERT on FewRel 1.0**
1-shot

|                   | Novel | Base | All |
|  ---------------  | -----------  | ------------- | ------------ |
| InreProtoNetwork   | 60.15 | 82.10 | 71.13 |
| ICA-Proto | 63.25 | 82.56 | 72.91 |
| Ours | 80.40 | 78.31 | 79.36 |

5-shot

|                      | Novel | Base | All |
|  ---------------  | -----------  | ------------- | ------------ |
| InreProtoNetwork   | 65.77 | 84.64 | 75.21 |
| ICA-Proto | 69.49 | 84.89 | 77.19 |
| Ours | 83.39 | 82.67 | 83.03 |

Acknowledgment
Our code is based on the implementations of SimpleFSRE(https://github.com/lylylylylyly/SimpleFSRE)
