---
title: "Predicting Cell Effective Permeability with Gradient Boosted Regression"
excerpt: "The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue. <br/><img src='/images/Caco2/cell membrane.PNG' width='533' height='300'>"
collection: portfolio
---
# TDC ADMET, Caco-2_Wang Submission
The structure for this model was originally written by Lealia Xiong, and it is used with permission and attribution. Modifications including scaling the y values and tuning the hyperparameters were done by Rob Learsch. 

*Dataset Description*: The human colon epithelial cancer cell line, Caco-2, is used as an in vitro model to simulate the human intestinal tissue. The experimental result on the rate of drug passing through the Caco-2 cells can approximate the rate at which the drug permeates through the human intestinal tissue.

Task Description: Regression. Given a drug SMILES string, predict the Caco-2 cell effective permeability.

Dataset Statistics: 906 drugs. 


```python
from typing import Tuple

import numpy as np
import pandas as pd

# cheminformatics
import rdkit.Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# logging
import tqdm

# data preprocessing
import sklearn.impute
import sklearn.preprocessing

# modeling
import sklearn.ensemble

# metrics
import sklearn.metrics


```


```python
from tdc.single_pred import ADME
data = ADME(name = 'Caco2_Wang')
split = data.get_split()
```

    Found local copy...
    Loading...
    Done!


## Add features
The features used are the basic chemical descriptors from RDKit. 


```python
def add_descriptor_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Use rdkit to get descriptors of each drug in the `data` df.
    Return a Pandas DataFrame with the descriptors as columns in the df and .
    """
    
    # Extract the Drug column
    assert 'Drug' in data.columns, "'Drug' must be a column in the input DataFrame."
    drugs = data['Drug']
    y = data['Y']
    
    # Get the descriptors for each drug
    print("Calculating descriptors...")
    descriptors = []
    for drug, target in tqdm.tqdm(zip(drugs, y)):
        descriptor = Descriptors.CalcMolDescriptors(
            rdkit.Chem.MolFromSmiles(drug)
        )
        descriptor['Drug'] = drug
        descriptor['Y'] = target
        descriptors.append(descriptor)

    # Make a dataframe for the descriptors
    df = pd.DataFrame(descriptors)

    return df
```

Processing is done to impute values for the cells with missing values and to scale both the X-data and y-data separately. Be sure to only train the imputer and scalers on the training data - not the test data.


```python
def preprocess_data(
    data: pd.DataFrame, 
    imputer=sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean'),
    fit_imputer=True,
    scaler_X=sklearn.preprocessing.RobustScaler(),
    scaler_y=sklearn.preprocessing.RobustScaler(),
    fit_scaler=True
):
    """
    Imputes missing values.
    Scales feature data.

    Returns a tuple X, y of scaled feature data and target data.
    """

    col_array = np.array(data.columns)

    # extract just the feature data
    X = data[col_array[~np.isin(col_array, ['Drug_ID', 'Drug', 'Y'])]].to_numpy()
    
    # extract the target data
    y = np.array(data['Y']).reshape(-1,1)
    
    # impute missing data
    if imputer is not None:
        if fit_imputer:
            X = imputer.fit_transform(X)
        else:
            X = imputer.transform(X)

    # scale the feature data
    if scaler_X is not None:
        if fit_scaler:
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)
        else:
            X = scaler_X.transform(X)
            y = scaler_y.transform(y)



    return X, y, imputer, scaler_X, scaler_y
```


```python
regr = sklearn.ensemble.GradientBoostingRegressor()
X_train, y_train, imputer, scaler_X, scaler_y = preprocess_data(add_descriptor_columns(split['train']))
regr.fit(X_train, y_train)
```

    Calculating descriptors...


    637it [00:05, 111.49it/s]





<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GradientBoostingRegressor<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor()</pre></div> </div></div></div></div>




```python
y_train_pred = regr.predict(X_train)

sklearn.metrics.mean_absolute_error(y_train, y_train_pred)
```




    0.14833562767542




```python
X_val, y_val, _, _,_ = preprocess_data(
    add_descriptor_columns(split['valid']),
    imputer=imputer, fit_imputer=False,
    scaler_X=scaler_X, scaler_y=scaler_y,
    fit_scaler=False
)
```

    Calculating descriptors...


    91it [00:00, 125.23it/s]



```python
y_val_pred = regr.predict(X_val)

sklearn.metrics.mean_absolute_error(y_val, y_val_pred)
```




    0.2834360140217522




```python
X_test, y_test, _, _, _ = preprocess_data(
    add_descriptor_columns(split['test']),
    imputer=imputer, fit_imputer=False,
    scaler_X=scaler_X, scaler_y=scaler_y,
    fit_scaler=False
)
```

    Calculating descriptors...


    182it [00:01, 119.27it/s]



```python
y_test_pred = regr.predict(X_test)
sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
```




    0.2656885538922705



Not a bad start. Let's dial in the hyperparameters and improve it a bit more.

## Tune hyperparameters:
Use the validation dataset to test the hyperparameters of the model. The learning rate and number of estimators help the model to learn from the training set. The subsample helps to prevent over-fitting by randomly removing some training data when fitting and produces a more general model, which should be better at predicting the results in the validation and test datasets.

I'm using a homemade version of GridSearchCV to find the parameters because I already have a validation set defined.


```python
from sklearn.model_selection import ParameterGrid
```


```python
params = [    
    {'subsample' : np.linspace(0.5,0.9,5),
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], 
    'n_estimators': [100, 125, 150, 175, 200, 225, 250,],
    }
]
```


```python
best_score = sklearn.metrics.mean_absolute_error(y_val, y_val_pred)
best_set = {}
for param_set in ParameterGrid(params):
    regr = sklearn.ensemble.GradientBoostingRegressor(
    random_state=1,
    )
    regr.set_params(**param_set)
    regr.fit(X_train,y_train)
    score_MAE = sklearn.metrics.mean_absolute_error(y_val, regr.predict(X_val))
    #lower is better
    # save if best
    if score_MAE < best_score:
        best_score = score_MAE
        best_set = param_set
```


```python
from tdc.benchmark_group import admet_group
group = admet_group(path = 'data/')
predictions_list = []
for seed in [1, 2, 3, 4, 5]:
    benchmark = group.get('Caco2_Wang') 
    predictions = {}
    name = benchmark['name']
    train, test = benchmark['train_val'], benchmark['test']
    print(f"Seed {seed}:")
    X_train, y_train, imputer, scaler_X, scaler_y = preprocess_data(add_descriptor_columns(train))
    X_test, y_test, _, _, _ = preprocess_data(
        add_descriptor_columns(test), 
        imputer=imputer, fit_imputer=False, 
        scaler_X=scaler_X, fit_scaler=False, 
        scaler_y=scaler_y
    )
    
    regr = sklearn.ensemble.GradientBoostingRegressor(
        random_state=seed,
    )
    regr.set_params(**best_set)
    regr.fit(X_train, y_train)

    y_pred_test_scaled = regr.predict(X_test)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1,1))
    y_pred_test = y_pred_test.reshape(-1)
    
    predictions = {}
    prediction_dict = {name: y_pred_test}
    predictions_list.append(prediction_dict)
    
results = group.evaluate_many(predictions_list)
print(results)
```

    Found local copy...


    Seed 1:
    Calculating descriptors...


    728it [00:06, 115.24it/s]


    Calculating descriptors...


    182it [00:01, 113.16it/s]


    Seed 2:
    Calculating descriptors...


    728it [00:06, 114.21it/s]


    Calculating descriptors...


    182it [00:01, 111.07it/s]


    Seed 3:
    Calculating descriptors...


    728it [00:06, 114.39it/s]


    Calculating descriptors...


    182it [00:01, 111.53it/s]


    Seed 4:
    Calculating descriptors...


    728it [00:06, 114.49it/s]


    Calculating descriptors...


    182it [00:01, 111.84it/s]


    Seed 5:
    Calculating descriptors...


    728it [00:06, 114.72it/s]


    Calculating descriptors...


    182it [00:01, 110.93it/s]


    {'caco2_wang': [0.274, 0.004]}


As of 2024/07/21, this is a first place score on the [leaderboard](https://tdcommons.ai/benchmark/admet_group/01caco2/)!

The current leader is MapLight, Jim Notwell, 0.276 ± 0.005 


```python
print('Number of parameters: '+str(len(regr.get_params())))
```

    Number of parameters: 21



```python
regr.get_params()
```




    {'alpha': 0.9,
     'ccp_alpha': 0.0,
     'criterion': 'friedman_mse',
     'init': None,
     'learning_rate': 0.06,
     'loss': 'squared_error',
     'max_depth': 3,
     'max_features': None,
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 250,
     'n_iter_no_change': None,
     'random_state': 5,
     'subsample': 0.5,
     'tol': 0.0001,
     'validation_fraction': 0.1,
     'verbose': 0,
     'warm_start': False}

