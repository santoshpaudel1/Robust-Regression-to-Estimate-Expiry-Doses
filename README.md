# Robust-Regression-to-Estimate-Expiry-Doses

**Robust-Regression-to-Estimate-Expiry-Medicine-Doses* implements ML algorithm to estimate the doses that will expire over fixed date (30, 60, 90, 120 )days for a given medicine in a medstation(Pyxis device). In previous linear regression result was far away form ground truth. Due to the spairsity present in removal counts in pockets, my contribution on this project is to replace the original linear regression by robust reggression (Hubers) mothod in order to take a consideration of the sparsity (inconsistency) so that the prediction by regression method is more closer to the ground truth.
In this project:
- Robust linear regression (Huber’s) method on big data environment (IOA/MKP
hospital database).
- Compare the estimation accuracy with respect to original statistical method (heuristic method) and previous estimation method by using linear regression. 
- Results shows that prediction accuracy improved by at least 4.5% as compared to original regression model and at least
10% as compared to heuristic model (Test/ Validation on 43 different hospital pharmacy sites).


## Training/ Testing Datasets
The IOA/ MKP database is used for datasets. The data are stored in distributed database n MPP data platform (Impala). The SQL querry used  for data extraction, feature engineering is in seperate folder "SQL_data_extraction" of the project
## Installation/ Run Instruction

1. Go to the directory of the project folder
2. Run python STE_script.py data/ results/ 2

	
 ## Folder Structure Conventions

#### A top-level directory layout

```
├── SQL_data_extraction               # contain SQL queries for input data preparation and features extraction
│             
├── STE_notebook_code                 # Preprocess, implementation of Robust Hubers Regression in Jupyter notebook       
│
├── STE_python_code                   # Implement Robust Hubers Regression in python 
│   ├── data                          # Input datasets
│   │── results                       # Folder to save the results for each hospital sites
│   │── STE_hybrid_funcs.py           # List of all functions used for the prediction (training/ testing)
    │── STE_hybrid_utilities.py       # Function for display results (.csv format), box plots for each sites
    │── STE_script.py                 # Main code for training and testing for all sites
 

