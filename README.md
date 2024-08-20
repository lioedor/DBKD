# DBKD
## Dataset
* [MIMIC-III](https://mimic.mit.edu/): The database comprises detailed clinical information regarding >60 000 stays in ICUs at the Beth Israel Deaconess Medical Center in Boston between 2001 - 2012. MIMIC Code Repository: https://github.com/MIT-LCP/mimic-code.
* BHF dataset: 4,899 Chinese EMRs from Beijing Friendship Hospital between January 2020 and January 2022. The BHF dataset will be made available from the corresponding author by request.

## Requirement
* Python==3.6
* Pytorch==0.4.1
* CUDA==9.0
* Scikit-learn==0.23.2
* Numpy==1.16.5
* Scipy==1.3.1
* Pandas==0.23.4
* Matplotlib==3.3.2

## Baselines
* [CAML](https://github.com/jamesmullenbach/caml-mimic)
* [HyperCore](https://aclanthology.org/2020.acl-main.282/)
* [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network)
* [MSATT-KG](https://dl.acm.org/doi/abs/10.1145/3357384.3357897)
* [JAN](https://ieeexplore.ieee.org/document/9822203)

## Data usage
Put the files of MIMIC III into the 'data' dir as below:
