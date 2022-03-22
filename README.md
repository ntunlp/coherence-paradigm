# Rethinking Self-Supervision Objectives for Generalizable Coherence Modeling
Code and Data for the ACL 2022 paper: https://arxiv.org/abs/2110.07198

### Pre-requisites
```
* Python=3.6
* Pytorch>=1.10.1
* Huggingface Transformers=4.13
```

### Input Data Format

- All models require train, dev and test files in pickle format as input. The specific format is:
    - The pickle file should be a list of dictionaries. Each dictionary has two keys, 'pos' and 'negs' (or 'neg' for pairwise data). 'pos' should contain the list of sentences from the positive or coherent document, while 'negs' should contain the list of negative documents (e.g. incoherent documents, permutations) which are in turn lists of sentences.
    - The `--data_type` argument should be set to `single` or `multiple` depending on the number of negatives in the dataset.
        - e.g. `single`: [{'pos':['aa', 'bb', 'cc'], 'neg':['bb', 'aa', 'cc']}..{}]
        - e.g. `multiple`: [{'pos':['aa', 'bb', 'cc'], 'negs':[['bb', 'aa', 'cc'], ['cc', 'aa', 'bb'], ['bb', 'cc', 'aa']]}..{}]

### Training the model
Navigate into the model folder that you want to train (pairwise, contrastive or our full hard negative model with momentum encoder). 
```
> CUDA_VISIBLE_DEVICES=x python train.py --train_file [train.pkl] --dev_file [dev.pkl]
```
Please refer to the `args.py` file for all other arguments that can be set. All hyperparameter defaults are set to the values used for experiments in the paper.
       
### Evaluation using the Coherence Model
You can use our trained model to evaluate machine generated text. More details will be updated soon.

