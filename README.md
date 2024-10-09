# How to obtain SF36-scores in different subdomains

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10086861.svg)](https://doi.org/10.5281/zenodo.10086861)

Run the soft from terminal using ```python3 sf36.py [-d input_path] [-o output_path] [-ind]```  

- `[-d]` optional, defines the path to your data. Default is [~/data/](https://github.com/MatthieuGG/SF36-scores/tree/main/data) in the same folder.  
- `[-o]` optional, defines the path to your results. Default is [~/results/](https://github.com/MatthieuGG/SF36-scores/tree/main/results) in the same folder.  
- `[-ind]` optional, saves individual files. Default is [one concatenated file](https://github.com/MatthieuGG/SF36-scores/blob/main/results/concatenated_results.csv).  

You first have to go in the folder where sf36.py is located using `cd`. Exemple of use: 

```
> cd /Users/Me/Downloads/SF36-scores-main/  
> python3 sf36.py -d /Users/Me/Documents/sf36/myData/ -o /Users/Me/Documents/sf36/myResults/ -ind
```

See more information in the [documentation]() *(not yet edited)*. **WARNING: we use a version of the SF36 in which the order or the questions are different from original.** See an exemple here [SF36.pdf]() *(not provided yet)* to check your questions order. We used the guidelines  ["SF36 Health Survey - Manual and Interpretation Guide" from John E. Ware, Jr. PhD](https://www.researchgate.net/publication/247503121_SF36_Health_Survey_Manual_and_Interpretation_Guide).

**To cite this work:**
> Matthieu Gallou-Guyot. (2023). SF36-scores. Zenodo. https://doi.org/10.5281/zenodo.10060411 

*(old version using a Jupyter Nootebook is still avaliable in [~/old/](https://github.com/MatthieuGG/SF36-scores/tree/main/old))*
