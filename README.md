# How to obtain SF36-scores in different subdomains

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10086861.svg)](https://doi.org/10.5281/zenodo.10086861)

**Input: individual .csv files  ([~/sample](https://github.com/MatthieuGG/SF36-scores/tree/main/sample))  
Output: individual and comon .csv files  ([~/results](https://github.com/MatthieuGG/SF36-scores/tree/main/results))  
Script: [Scoring SF36.ipynb](https://github.com/MatthieuGG/SF36-scores/blob/main/Scoring%20SF36.ipynb)**  

SF36 questionnaires assess the quality of life (QoL). Through 36 items, you can obtain sub-domain (Physical Functioning,	Role-Physical,	Bodily-Pain,	General Health,	Vitality,	Social Functioning,	Role-Emotional,	Mental Health,	Reported Health Transition,	Mean Current Health) as well as general domains (PHYSICAL,	MENTAL,	GLOBAL) 0 to 100 scores.  

To do so, you need to:
* transpose your paper/pdf/online questionnaires into .csv files, and put them all in the same folder (see exemple file structure in [~/sample](https://github.com/MatthieuGG/SF36-scores/tree/main/sample)) **WARNING: we use a version of the SF36 in which the order or the questions are different from original. Please check you use the same question order.**
* use the code provided (**[Scoring SF36.ipynb](https://github.com/MatthieuGG/SF36-scores/blob/main/Scoring%20SF36.ipynb)**) on your transposed .csv files to obtain the scores (see [~/results](https://github.com/MatthieuGG/SF36-scores/tree/main/results)). You can run the code as is if you keep the same path.  
* This notebook imports the data, check for conditions (duplicates, missing data, out of range, reorganise), and calculates the different subdomains values, then saves the results.
---
We used the guidelines  ["SF36 Health Survey - Manual and Interpretation Guide" from John E. Ware, Jr. PhD](https://www.researchgate.net/publication/247503121_SF36_Health_Survey_Manual_and_Interpretation_Guide).

**To cite this work:**
> Matthieu Gallou-Guyot. (2023). SF36-scoring. Zenodo. https://doi.org/10.5281/zenodo.10060411 