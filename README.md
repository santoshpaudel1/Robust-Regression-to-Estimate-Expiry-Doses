# Robust-Regression-to-Estimate-Expiry-Doses

**Robust-Regression-to-Estimate-Expiry-Medicine-Doses* implements ML algorithm to estimate the doses that will expire over fixed date (30, 60, 90, 120 )days for a given medicine in a medstation(Pyxis device). In previous linear regression result was far away form ground truth. Due to the spairsity present in removal counts in pockets, my contributi0n on this project is to replace the original linear regression by robust reggression (Hubers) mothod in order to take a consideration of the sparsity (inconsistency) so that the prediction by regression method is more closer to the ground truth.
In this project:e 
- Robust linear regression (Huber’s) method on big data environment (IOA/MKP
hospital database).
- Predict potential CKD subtypes

## Installation Instruction

### Window 
1. conda create -n test python=3.6.7 pip
2. conda activate test
3. pip3 install -r requirements.txt (All reqiured packages are listed on requirements text file)
	```scikit-learn==0.23.0
	flask==1.0.2, scikit-learn==0.23.0,numpy==1.18.1, pandas==1.1.4, matplotlib==3.0.2...

4. go the project main folder (CKD_project) directory 
	```bash
	$> cd. ...\CKD_project_file\CKD_project
	```
5. train the model (train/test results will be saved in respective folder) 
	```bash
	$> ...\CKD_project_file\CKD_project>python train_model.py
	```
6. run python main.py 
	```bash
	$> ...\CKD_project_file\CKD_project>python main.py
	```
7. open  http://localhost:8080/
8. enter the features input on web API and press 'predict' button 

	
 ## Folder Structure Conventions

#### A top-level directory layout

```
├── input
│   ├── data                      # raw data
│   └── processed_data            # processed and semiprocessed data for feaures extraction
├── model                         # contain files related trained/ tested model in .pkl and .csv form
│
├── templates                     # flask/ HTLM templetes for UI
│   ├── form.html
│   │── main.html
│   │── result.html
│
├── utility_func                 # functions for data preprocessing, training and testing 
│   ├── preprocessing.py
│   └── utility.py
│
├──CKD_train_model.ipynb         # premilinary analysis and data pre preprocessing in jupyter notebook
│
├── main.py                      # main code to run the saved trained model and UI 
│
├── train_model.py               # trained the model and save it for main code
    
 
## Reference
-I used some format for html API from https://github.com/venkata-sreeram/Chronic-Kidney-Disease-Prediction

