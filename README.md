# On End-to-End White-Box Adversarial Attacks <br> in Music Information Retrieval
This is the code for the accepted [TISMIR](https://transactions.ismir.net/) submission
*On End-to-End White-Box Adversarial Attacks in Music Information Retrieval* (Institute for Computational 
Perception, JKU Linz). To view our supplementary material, including listening examples, 
look at our [Github page](https://cpjku.github.io/adversaries_in_mir/). 

Note that this work is an extension of our previous technical report [End-to-End Adversarial White Box Attacks on 
Music Instrument Classification](https://arxiv.org/abs/2007.14714), and therefore includes the 
code for reproducing the experiments we presented there as well.

This work is supported by the Austrian National Science Foundation (FWF, P 31988).

## Prerequisites

- [pytorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/)
- [numpy](https://pypi.org/project/numpy/)
- [attrdict](https://pypi.org/project/attrdict/)
- [matplotlib](https://matplotlib.org/)
- [parse](https://pypi.org/project/parse/)
- [librosa](https://librosa.org)
- [h5py](https://pypi.org/project/h5py/)

For details on the versions of the libraries we used, please 
view `requirements.txt`. The tested Python version is `3.7.3`.
If `conda` is available, a new environment
can be created and the necessary libraries installed via

````
conda create -n ad_env python==3.7.3 pip
conda activate ad_env
pip install -r requirements.txt
````

## Attacking an Instrument Classifier

#### Data

The data we use is a part of the curated train set of the 
FSDKaggle2019 [1, 2] data. More precisely,
we use the 799 files that have one single of 12 different musical labels
(see `instrument_classifier/data/data_utils`) for more information on these labels.

To be able to run this code, download the [curated data](https://zenodo.org/record/3612637), 
and make sure to set the `d_path` in `instrument_classifier/utils/paths.py` 
correctly, pointing to the extracted directory. 
After downloading the data, you might want to re-sample it to 16kHz
(e.g. with `torchaudio.transforms.Resample`), as we used 16kHz to train
the given pre-trained model.
Additionally, we need the labels which are available in 
`train_curated_post_competition.csv` 
(and `test_post_competition.csv`); for this, please set the 
`csv_path` to point to these files.

[1] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, 
 Xavier Serra (2020). FSDKaggle2019 (Version 1.0) [Data set]. 
Zenodo. http://doi.org/10.5281/zenodo.3612637.

[2] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, 
Xavier Serra. "Audio tagging with noisy labels and minimal supervision". 
Proceedings of the DCASE 2019 Workshop, NYC, US (2019).

#### Training a network
To train a network, first define the parameters in `instrument_classifier/training/params.txt`. Afterwards you can start training the network by running the following file:

````
python -m instrument_classifier.training.train 
````

#### Attacking the network

The folder `misc/pretrained_models` contains two pre-trained models we use for our experiments 
(use model `torch16s1f_2` for extended experiments, cf. section A.5.2 in the supplementary material). 

*FGSM*: To run this untargeted attack, define parameters in 
`instrument_classifier/baselines/fgsm_config.txt`. Then, you can run the 
python script, e.g. with

````
python -m instrument_classifier.baselines.fgsm
````

*PGDn*: The second untargeted attack can be run similarly. First,
define parameters in `instrument_classifier/baselines/pgdn_config.txt`, and
run the python script from command line with e.g.

````
python -m instrument_classifier.baselines.pgdn
````

*C&W* and *Multi-Scale C&W*: Both targeted attacks can be performed by running
````
python -m instrument_classifier.attacks.targeted_attack
````

To modify the parameter setting, change the according parameters in 
`instrument_classifier/attacks/attack_config.txt`. You can
switch between the two attacks by setting `attack = cw` or `attack = mscw`
respectively; in addition to that, the target class can be modified to
be either `target = random` or the name of a particular classs, e.g.
`target = accordion`.

#### Defence
To test the FGSM and PFD attack on the defence networks run the following code:

````
python -m instrument_classifier.defence.defence --n_def_nets=10 --csv_name=final_defence_valid_false
````

#### Evaluation

In order to evaluate your experiments, you can make use of 
functions provided in the `instrument_classifier/evaluation` directory:

- `confusion_matrices.py` allows you to plot confusion matrices;
- and `eval_funcs.py` contains methods to compute accuracies, the SNR, and iterations.

## Attacking a Music Recommender

#### Data and System

The system we perform our experiments on is the [Fm4 Soundpark](https://fm4.orf.at/soundpark/) [3], 
where various songs can be found and downloaded. We cannot provide the exact data we used 
in this work here, nevertheless the code should provide a good starting base for experiments.
For more information, please contact us directly.

[3] Martin Gasser and Arthur Flexer (2009). Fm4 soundpark: Audio-based music recommendation in everyday use. 
In Proc. of the 6th Sound and Music Computing Conference, SMC, pages 23–25.

#### Experiments

Before running experiments, you should adapt the paths to your own in `recommender/utils/paths.py`.
We additionally prepare our data by storing it with the HDF5 binary data 
format (see `recommender/data/preprocess.py`).

To run the adversarial attacks, and increase the number of times a particular song is recommended,
you can use both

````
python -m recommender.attacks.kl_min
python -m recommender.attacks.kl_min_approx
````

Here the first command runs the standard attack, and the second one a less complex version, in which the
k-occurrence is first approximated to speed up the convergence checks if a large number of files is used.

#### Evaluation

For evaluation of the attacks, we provide a few files as well in `recommender/evaluation`:

- `log_processing.py` contains methods to process log-files, returning things such as SNRs, 
k-occurrences or how much files converged;
- `noisy_test` contains a script which allows us to check whether we can 
promote songs to 'hubs' by adding white noise with a specified SNR;
- and `plot_evals` allows to make a plot contrasting k-occurrences before and after an attack.
