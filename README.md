

##  Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts
This repo contains a pytorch implementation for the paper: Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts
### Setups
The code requires:
- python 3.8
- pytorch 2.0.1+cu118

Please see *requirements.txt* for other required packages.

### Datasets

For meta-training and meta-testing on the Meta-dataset benchmark, please follow the instruction in [PMF](https://github.com/hushell/pmf_cvpr22?tab=readme-ov-file#meta-dataset) first to download the datasets in .h5 format.

### Experiments
#### Before meta-training
> `export mdh5_path=PATH/TO/DOWNLOADED_METADATASET`\
> `ulimit -n 50000` 
> 


#### For meta-training 
> 
> `python -m src.trainer --yaml md_smat --seed 0`

- The full lists of hyperparameters used for the experiments can be found in the *config* directory. 
- Your runs / training logs / testing results will all be saved in a directory called _experiments_ by default. To change this default option, please modify the _logging_ variables in the config.yaml file. 

#### For meta-testing 

- no fine-tuning i.e., inference using prototnet
>
> ` python -m src.tests.md_few_shot --tasks md --alg smat --n 600 --i filename_to_log_your_results --checkpoint path/to/checkpoint.pt`
>
- fine-tuning with {full, lora}
>
> `python -m src.tests.md_few_shot --tasks md --hps --aug --alg smat --n 600 --i filename_to_log_your_results --finetune_mode {full, lora} --checkpoint path/to/checkpoint.pt`
>


### Citation
If you find this repository useful in your research, please consider citing the following paper:


### Contact
If you have any question please feel free to **Contact** Shengzhuang Chen **Email**: szchen9-c [at] my [dot] cityu [dot] edu [dot] hk  