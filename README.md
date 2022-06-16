## VF-iDCA
This repository provide a realization of VF-iDCA on hyperparameter selection problems in Python with codes for reproduce the numerical result of experiments and a general function.

The algorithm and the models are presented in the paper [_Value Function Based Difference-of-Convex Algorithm for Bilevel Hyperparameter Selection Problems_](https://arxiv.org/pdf/2206.05976.pdf)

## Dependencies
Based on anaconda (`numpy` for calculation, `pandas` for saving data and `matplotlib` for plotting) and `cvxpy` for defining and solving optimization problems.

To run the experiments for comparison, you should also install [`hyperopt`](https://github.com/hyperopt/hyperopt) (for Bayes method).

The code for synthetic data generation and experiments on IGJO algoritm are based on this [repo](https://github.com/jjfeng/nonsmooth-joint-opt). 

IFDM algorithm is based on [`sparse_ho`](https://github.com/QB3/sparse-ho) with little modification to make sure the data was shared.

## Usage

### For Experiments
- Clone this repository
- Run the following command under the experiments folder
    ```
    python ElasticNet_Experiments.py
    ```
    You could change the parameters in the python file according to your demand.

### For General Usage
`VF_iDCA.py` provides a framework of our alogirthm, however, you need to write the model on your own as `wlasso.py` does, programming details are recorded in `tutorial_wLasso.ipynb`.

`wlasso.py` provides an example on using VF-iDCA solve hyperparameter selection problem on wlasso model.

## Cite
If you use this code, please cite:
```
@article{gao2022difference,
  title={Value Function Based Difference-of-Convex Algorithm for Bilevel Hyperparameter Selection Problems},
  author={Gao, Lucy and Ye, Jane J. and Yin, Haian and Zeng, Shangzhi and Zhang, Jin},
  journal={arXiv preprint arXiv:2206.05976},
  year={2022}
}
```
