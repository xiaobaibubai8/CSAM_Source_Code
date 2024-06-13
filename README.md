# NCFG-CS_Source_Code

This is the source codes for the NCFG-CS model and other baseline model 
The Source code is coming soon.



## Introduction

This is the deeplearning project for code search.We use Pytorch to implement the model.



## Environment dependencies

You can use the following commands to install these dependencies
```bash
pip install -r requirements.txt
```


## Dataset

We use the CodeSearchNet dataset, and the preprocessed dataset can be accessed via the cloud drive 
(link)

If you want to quickly start training/testing the model, you can directly obtain ther pocessed data files preprocessed_train/valid/text.bin and place them in the /NCFG-CS_Source_Code/NCFG-CS/preprocessed_data/CSN/train(/valid/test)/processed folder.



## models

Our pre-trained models and the corresponding logs are located in /NCFG-CS_Source_Code/NCFG-CS/saved_model.



## train model 

Use the following commands to train the model
```bash
python train.py -batch_size 128 -num_epoch 100 -emb_size 128 -hidden_dim 256 -pool_size 2000 -loss_type cos -saved_model_dir ./saved_model/model1 -train
```

## test model

Use the following commands to test the model
```bash
python train.py -batch_size 32 -num_epoch 100 -emb_size 128 -hidden_dim 256 -pool_size 2000 -loss_type cos -pool_size 2000 -test -test_model_path ./saved_model/model1/best_model.bin
```
> ./saved_model/model2/record.log


## baseline models

The rest of the model code can be referred to in the link.
[baseline code](https://gitee.com/weicc214/code_graph_nn_pyg)







