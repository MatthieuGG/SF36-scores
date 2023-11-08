# How to obtain SF36-scores


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10060412.svg)](https://doi.org/10.5281/zenodo.10060412)


SF36 questionnaires assess the quality of life (QoL). Through 36 items, you can obtain sub-domain (Physical Functioning,	Role-Physical,	Bodily-Pain,	General Health,	Vitality,	Social Functioning,	Role-Emotional,	Mental Health,	Reported Health Transition,	Mean Current Health) as well as general domains (PHYSICAL,	MENTAL,	GLOBAL) 0 to 100 scores.  

To do so, you need to:
* transpose the questionnaires into .csv files (see the model provided in /sample/ SF36 - Sample.csv) **WARNING: we use a version of the SF36 in which the order or the questions are different from original**
* put all the .csv files in the same folder
* use the code provided (**Scoring SF36.ipynb**) on your transposed .csv files to obtain the scores. You have to precise the path of the folder where all the .csv are, and the path where you want to save the result.

We used the guidelines  "SF36 Health Survey - Manual and Interpretation Guide" from John E. Ware, Jr. PhD.
