This directory contains files to train TEIM-Seq and TEIM-Res.

## Requirements
Same as the environment for inference (requirement file is provided [here](../requirements.txt) in the parent directory).

## Train TEIM-Seq
In this directory, run the following command to train TEIM-Seq using cross-validation:
```python
python scripts/train_seqlevel.py --config configs/seqlevel_cv_shuffle.yml
```
Another configuration file `configs/seqlevel_all.yml` is provided for training TEIM-Seq with all sequence-level data.

## Train TEIM-Res
In this directory, run the following command to train TEIM-Res using cross-validation:
```python
python scripts/train_reslevel.py --config configs/<config_file>
```
where `<config_file>` is one of the following:
- reslevel_bothnew.yml: split with both-new setting (both CDR3 and epitope in the validation set are unseen in the training set)
- reslevel_newcdr3.yml: split with new-CDR3 setting
- reslevel_newepi.yml: split with new-epitope setting

