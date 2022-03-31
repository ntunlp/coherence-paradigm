# Rethinking Self-Supervision Objectives for Generalizable Coherence Modeling
Code and data for the ACL 2022 paper: https://arxiv.org/abs/2110.07198

### Pre-requisites
```
* Python=3.6
* Pytorch>=1.10.1
* Huggingface Transformers=4.13
```

### Input Data Format

- All models require train, dev and test files in pickle format as input. The specific format is:
    - The pickle file should be a list of dictionaries. Each dictionary has two keys, `pos` and `negs` (or `neg` for pairwise data). 
    - `pos` should contain the list of sentences from the positive or coherent document
    - `negs` should contain the list of negative documents (e.g. incoherent documents, permutations) which are in turn lists of sentences.
    - The `--data_type` argument should be set to `single` or `multiple` depending on the number of negatives in the dataset.
        - e.g. `single`: [{'pos':['sentence_1', 'sentence_2', 'sentence_3'], 'neg':['sentence_2', 'sentence_1', 'sentence_3']}..{}]
        - e.g. `multiple`: [{'pos':['sentence_1', 'sentence_2', 'sentence_3'], 'negs':[['sentence_3', 'sentence_2', 'sentence_1'], ['sentence_1', 'sentence_3', 'sentence_2'], ['sentence_2', 'sentence_3', 'sentence_1']]}..{}]
       
You can look at the test sets provided in the `independent_test_sets` folder for the format.

### Evaluation using the Coherence Model
Download the pretrained model from: https://www.dropbox.com/sh/2q5s71zxc3o0tp6/AAA1TXbdR_xVBNSKXDkahFqma?dl=0. Navigate into the model folder you want to evaluate with (ensure this matches the pretrained model you downloaded). Run:
```
> CUDA_VISIBLE_DEVICES=x python eval.py --test_file [test.pkl] --data_type [single,multiple] --pretrained_model [model.pt]
```
The code will provide the accuracy calculated as the number of times the positive document was scored higher than the negative document. If you are comparing generated texts from two models, you can assign any of the models' outputs as `pos` and `neg` consistently and obtain the percentage of times the model designated `pos` is more coherent than the model designated `neg`. Simply subtract the accuracy from 100 to get the vice-versa value.

You can also pass any of the test files in the `independent_test_sets` folder or provide your own test set based on the format described above. In case the comparison is not so straight-forward (for e.g., the `LMvLM` dataset), the code also saves the scores in a pickle dump called `temp-eval-dump`. 

To evaluate the Krippendorff's alpha agreement for the `LMvLM` dataset, run the test set to save the scores first, and then run:
```
> python model_agreement.py LMvLM_Annotations.pkl temp-eval-dump
```

### Training the model
Navigate into the model folder that you want to train (pairwise, contrastive or our full hard negative model with momentum encoder). 
```
> CUDA_VISIBLE_DEVICES=x python train.py --train_file [train.pkl] --dev_file [dev.pkl]
```
Please refer to the `args.py` file for all other arguments that can be set. All hyperparameter defaults are set to the values used for experiments in the paper.

To evaluate the model on a test set, run
```
> CUDA_VISIBLE_DEVICES=x python eval.py --test_file [test.pkl] --data_type [single,multiple] --pretrained_model [saved_checkpoint.pt]
```



