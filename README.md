# Graph Information Aggregation Cross-Domain Few-Shot Learning for Hyperspectral Image Classification

Code for the paper: [Graph Information Aggregation Cross-Domain Few-Shot Learning for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/9812472).

<p align='center'>
  <img src='figure/Gia-CFSL.png' width="800px">
</p>

## Abstract

Most domain adaptation (DA) methods in cross-scene hyperspectral image classification focus on cases where source data (SD) and target data (TD) with the same classes are obtained by the same sensor. However, the classification performance is significantly reduced when there are new classes in TD. In addition, domain alignment, as one of the main approaches in DA, is carried out based on local spatial information, rarely taking into account nonlocal spatial information (nonlocal relationships) with strong correspondence. A graph information aggregation cross-domain few-shot learning **(Gia-CFSL)** framework is proposed, intending to make up for the above-mentioned shortcomings by combining FSL with domain alignment based on graph information aggregation. SD with all label samples and TD with a few label samples are implemented for FSL episodic training. Meanwhile, **intradomain distribution extraction block (IDE-block)** and **cross-domain similarity aware block (CSA-block)** are designed. The IDE-block is used to characterize and aggregate the intradomain nonlocal relationships and the interdomain feature and distribution similarities are captured in the CSA-block. Furthermore, feature-level and distribution-level cross-domain graph alignments are used to mitigate the impact of domain shift on FSL. Experimental results on three public HSI datasets demonstrate the superiority of the proposed method.

## Paper

Please cite our paper if you find the code or dataset useful for your research.

```
@ARTICLE{9812472,  
  author={Zhang, Yuxiang and Li, Wei and Zhang, Mengmeng and Wang, Shuai and Tao, Ran and Du, Qian},  
  journal={IEEE Transactions on Neural Networks and Learning Systems},   
  title={Graph Information Aggregation Cross-Domain Few-Shot Learning for Hyperspectral Image Classification},  
  year={2022},  
  volume={},  
  number={},  
  pages={1-14},  
  doi={10.1109/TNNLS.2022.3185795}}
```

## Requirements

CUDA Version: 10.2

torch: 1.6.0

Python: 3.6.5

## Dataset

The dataset directory should look like this:
```
datasets
├── Chikusei_imdb_128.pickle
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
└── paviaU
    ├── paviaU_gt.mat
    └── paviaU.mat
```

## Usage
Take Gia-CFSL method on the Chikusei (Source Data) and Pavia University (Target Data) as an example: 

1.Running the script `Chikusei_imdb_128.py` to generate preprocessed source domain data, where `patch_length = 4` is used for 9*9 patch szie.

2.Running 


### test2
Indian_pines
```
python  train_Gia-CFSL3.py --config config/Indian_pines.py
```


<details>
<summary>Commands for Other Datasets</summary>

Indian_pines
```
python train_Gia-CFSL.py --config config/Indian_pines.py
```
```
/home/XUPT/anaconda3/envs/py39tf/bin/python /home/XUPT/project/doing/15IEEE_TNNLS_Gia-CFSL-main/IEEE_TNNLS_Gia-CFSL-main/train_Gia-CFSL.py --config config/Indian_pines.py
```


paviaU
```
python train_Gia-CFSL.py --config config/paviaU.py
```
```
/home/XUPT/anaconda3/envs/py39tf/bin/python /home/XUPT/project/doing/15IEEE_TNNLS_Gia-CFSL-main/IEEE_TNNLS_Gia-CFSL-main/train_Gia-CFSL.py --config config/paviaU.py
```
salinas
```
python train_Gia-CFSL.py --config config/salinas.py
```
```
/home/XUPT/anaconda3/envs/py39tf/bin/python /home/XUPT/project/doing/15IEEE_TNNLS_Gia-CFSL-main/IEEE_TNNLS_Gia-CFSL-main/train_Gia-CFSL.py --config config/salinas.py
```


| Dataset | Model name         | transductive MRR| inductive MRR  |
| -------| ------------------ |---------------- | -------------- |
| NELL   | CSR-OPT            |     0.463       |      0.425     |
|        | CSR-GNN            |     0.577       |      0.511     |
| FB15K-237 | CSR-OPT            |     0.619       |      0.554     |
|           | CSR-GNN            |     0.781       |      0.624     |
| ConceptNet  | CSR-OPT            |     0.559       |      0.547     |
|             | CSR-GNN            |     0.606       |      0.611     |


</details>


3.`config/salinas.py` and `config/Indian_pines.py` are used for Salinas data and Indian Pines data.
 * `num_generation` denotes the cycle number of IDE-block.
 * `test_lsample_num_per_class` denotes the number of labeled samples per class for the target data.
 * `tar_input_dim` denotes the number of bands for the target data.

