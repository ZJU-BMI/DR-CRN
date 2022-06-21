DR-CRN
==
This is the code for implementing DR-CRN proposed in "Disentangled Representation for Sequential Treatment Effect Estimation" (under review). 

It is written in python 3.7 with numpy 1.18.1 and tensorflow 1.13.1.

The code of DR-CRN is built upon the Counterfactual Recurrent Network (CRN) work of Ioana Bica et al.(2020), https://github.com/ioanabica/Counterfactual-Recurrent-Network.
The dataset generation, hyper-parameter searching, network training and evaluation follow the procedures of CRN to ensure fair comparison.


Dataset
-
Dataset will be automatically generated when you run "run_DRCRN.py". Generated dataset will be saved in "results_dir"。

Example
-
To run parameter search and evaluate the results:

```python run_DRCRN.py --chemo_coeff=5 --radio_coeff=5 --model_name=DRCRN --b_encoder_hyperparm_tuning=True --b_decoder_hyperparm_tuning=True```
