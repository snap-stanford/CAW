# Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks

## Introduction

This is the reference PyTorch implementation of the paper:\
*Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks*.

The project website is: <https://snap.stanford.edu/caw/>


## Authors
[Yanbang Wang](https://cs.stanford.edu/~ywangdr), [Yen-Yu Chang](https://yuyuchang.github.io/), [Yunyu Liu](https://wenwen0319.github.io/), [Jure Leskovec](https://cs.stanford.edu/people/jure/), [Pan Li](https://sites.google.com/view/panli-uiuc/publications)

## Requirements
* `python >= 3.7`, `PyTorch >= 1.4`, please refer to their official websites for installation details.
* Other dependencies:
```{bash}
pandas==0.24.2
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
matploblib==3.3.1
numba==0.51.2
```
Refer to `environment.yml` for more details.


## Dataset and preprocessing
#### Option 1: Use our preprocessed data
We provide preprocessed datasets: Reddit, Wikipedia, Enron, and UCI. Download them from [here](https://drive.google.com/drive/folders/1umS1m1YbOM10QOyVbGwtXrsiK3uTD7xQ?usp=sharing) to `processed/`. Then run the following:
```{bash}
cd processed/
unzip data.zip
```
You may check that each dataset corresponds to three files: one `.csv` containing timestamped links, and two ``.npy`` as node & link features. Note that some datasets do not have node & link features, in which case the `.npy` files will be all zeros.

#### Option 2: Use your own data
Put your data under `processed` folder. The required input data includes `ml_${DATA_NAME}.csv`, `ml_${DATA_NAME}.npy` and `ml_${DATA_NAME}_node.npy`. They store the edge linkages, edge features and node features respectively. 

The `.csv` file has following columns
```
u, i, ts, label, idx
```
, which represents source node index, target node index, time stamp, edge label and the edge index. 

`ml_${DATA_NAME}.npy` has shape of [#temporal edges + 1, edge features dimention]. Similarly, `ml_${DATA_NAME}_node.npy` has shape of [#nodes + 1, node features dimension].


All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.

We also recommend discretizing the timestamps (`ts`) into integers for better indexing.
## Training Commands

#### Examples:

* To train **CAW-N-mean** with Wikipedia dataset in inductive training, sampling 64 length-2 CAWs every node, and with alpha = 1e-5:
```bash
python main.py -d wikipedia --pos_dim 108 --bs 32 --n_degree 64 1 --mode i --bias 1e-5 --pos_enc lp --walk_pool sum --seed 0
```

* To train **CAW-N-attn** with UCI dataset in transductive mode, sampling 32 length-1 CAWs every node, with alpha = 1e-6, and using another random seed 123:
```bash
python main.py -d uci --pos_dim 100 --bs 32 --n_degree 32 --n_layer 1 --mode t --bias 1e-6 --pos_enc lp --walk_pool attn --seed 123
```

Detailed logs can be found in `log/`, a one-line summary of the evaluation result will also be written to `log/oneline_summary.log` upon completion.
 
## Usage Summary
```txt
usage: Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs
       [-h] [-d {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks}] [-m {t,i}]
       [--n_degree [N_DEGREE [N_DEGREE ...]]] [--n_layer N_LAYER] [--bias BIAS] [--agg {tree,walk}] [--pos_enc {spd,lp,saw}]
       [--pos_dim POS_DIM] [--pos_sample {multinomial,binary}] [--walk_pool {attn,sum}] [--walk_n_head WALK_N_HEAD]
       [--walk_mutual] [--walk_linear_out] [--attn_agg_method {attn,lstm,mean}] [--attn_mode {prod,map}]
       [--attn_n_head ATTN_N_HEAD] [--time {time,pos,empty}] [--n_epoch N_EPOCH] [--bs BS] [--lr LR] [--drop_out DROP_OUT]
       [--tolerance TOLERANCE] [--seed SEED] [--ngh_cache] [--gpu GPU] [--cpu_cores CPU_CORES] [--verbosity VERBOSITY]
```

## Optional arguments
```{txt}
  -h, --help            show this help message and exit
  -d {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks}, --data {wikipedia,reddit,socialevolve,uci,enron,socialevolve_1month,socialevolve_2weeks}
                        data sources to use, try wikipedia or reddit
  -m {t,i}, --mode {t,i}
                        transductive (t) or inductive (i)
  --n_degree [N_DEGREE [N_DEGREE ...]]
                        a list of neighbor sampling numbers for different hops, when only a single element is input n_layer
                        will be activated
  --n_layer N_LAYER     number of network layers
  --bias BIAS           the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is
                        uniform sampling
  --agg {tree,walk}     tree based hierarchical aggregation or walk-based flat lstm aggregation
  --pos_enc {spd,lp,saw}
                        way to encode distances, shortest-path distance or landing probabilities, or self-based anonymous
                        walk (baseline)
  --pos_dim POS_DIM     dimension of the positional embedding
  --pos_sample {multinomial,binary}
                        two practically different sampling methods that are equivalent in theory
  --walk_pool {attn,sum}
                        how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other
                        walk_ arguments
  --walk_n_head WALK_N_HEAD
                        number of heads to use for walk attention
  --walk_mutual         whether to do mutual query for source and target node random walks
  --walk_linear_out     whether to linearly project each node's
  --attn_agg_method {attn,lstm,mean}
                        local aggregation method, we only use the default here
  --attn_mode {prod,map}
                        use dot product attention or mapping based, we only use the default here
  --attn_n_head ATTN_N_HEAD
                        number of heads used in tree-shaped attention layer, we only use the default here
  --time {time,pos,empty}
                        how to use time information, we only use the default here
  --n_epoch N_EPOCH     number of epochs
  --bs BS               batch_size
  --lr LR               learning rate
  --drop_out DROP_OUT   dropout probability for all dropout layers
  --tolerance TOLERANCE
                        toleratd margainal improvement for early stopper
  --seed SEED           random seed for all randomized algorithms
  --ngh_cache           (currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously
                        calculated to speed up repeated lookup
  --gpu GPU             which gpu to use
  --cpu_cores CPU_CORES
                        number of cpu_cores used for position encoding
  --verbosity VERBOSITY
                        verbosity of the program output
```

## Acknowledgement
Our implementation adapts the code [here](https://drive.google.com/drive/folders/1GaH8vusCXJj4ucayfO-PyHpnNsJRkB78) as the code base and extensively adapts it to our purpose. We thank the authors for sharing their code.

## Cite us
If you compare with, build on, or use aspects of the paper and/or code, please cite us:
```text
@inproceedings{
wang2021inductive,
title={Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks},
author={Yanbang Wang and Yen-Yu Chang and Yunyu Liu and Jure Leskovec and Pan Li},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=KYPz4YsCPj}
}
```

