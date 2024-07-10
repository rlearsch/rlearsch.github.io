---
title: "Plasma Protein Binding Rate"
excerpt: "Regression based on a dataset of 1,614 drugs from AstraZeneca. <br/> The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins in the blood. This rate strongly affect a drug's efficiency of delivery. The less bound a drug is, the more efficiently it can traverse and diffuse to the site of actions. From a ChEMBL assay deposited by AstraZeneca. <br/><img src='/images/protein-peptide/thumbnail.PNG' width='533' height='300'>"
collection: portfolio
---
```python
from tdc.benchmark_group import admet_group
from tdc.single_pred import ADME
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem 
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import GradientBoostingRegressor

```


```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

import sklearn.impute

```


```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.neural_network import MLPRegressor

```


```python
from tdc.single_pred import ADME
data = ADME(name = 'PPBR_AZ')
split = data.get_split()
training, test, valid = split['train'], split['test'], split['valid']
# Training, test, and validiation datasets
```

    Found local copy...
    Loading...
    Done!


## Feature Engineer
One set of features are the Morgan Fingerprints from RDKit, another is the chemical descriptors - the chemical physical properties.


```python
# First, convert smiles strings to molecular information
test['mol'] = test['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
training['mol'] = training['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
valid['mol'] = valid['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
```


```python
y_train, y_valid, y_test = training['Y'].to_frame(), valid['Y'].to_frame(), test['Y'].to_frame()
```

### Chemical properties
From RDKit, built it


```python
X_test_chem = [Descriptors.CalcMolDescriptors(x) for x in test['mol']]
X_train_chem = [Descriptors.CalcMolDescriptors(x) for x in training['mol']]
X_valid_chem = [Descriptors.CalcMolDescriptors(x) for x in valid['mol']]

```


```python
pd.DataFrame(X_train_chem).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MaxAbsEStateIndex</th>
      <th>MaxEStateIndex</th>
      <th>MinAbsEStateIndex</th>
      <th>MinEStateIndex</th>
      <th>qed</th>
      <th>SPS</th>
      <th>MolWt</th>
      <th>HeavyAtomMolWt</th>
      <th>ExactMolWt</th>
      <th>NumValenceElectrons</th>
      <th>...</th>
      <th>fr_sulfide</th>
      <th>fr_sulfonamd</th>
      <th>fr_sulfone</th>
      <th>fr_term_acetylene</th>
      <th>fr_tetrazole</th>
      <th>fr_thiazole</th>
      <th>fr_thiocyan</th>
      <th>fr_thiophene</th>
      <th>fr_unbrch_alkane</th>
      <th>fr_urea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.901867</td>
      <td>12.901867</td>
      <td>0.237428</td>
      <td>-0.861281</td>
      <td>0.435219</td>
      <td>11.968750</td>
      <td>452.902</td>
      <td>431.734</td>
      <td>452.136366</td>
      <td>164</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12.059240</td>
      <td>12.059240</td>
      <td>0.178128</td>
      <td>-1.028493</td>
      <td>0.687391</td>
      <td>12.000000</td>
      <td>307.390</td>
      <td>282.190</td>
      <td>307.178358</td>
      <td>122</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12.477711</td>
      <td>12.477711</td>
      <td>0.063209</td>
      <td>-0.063209</td>
      <td>0.566619</td>
      <td>16.757576</td>
      <td>452.515</td>
      <td>424.291</td>
      <td>452.217203</td>
      <td>174</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.089598</td>
      <td>12.089598</td>
      <td>0.304671</td>
      <td>-0.558346</td>
      <td>0.651154</td>
      <td>14.931034</td>
      <td>430.339</td>
      <td>409.171</td>
      <td>429.112316</td>
      <td>150</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.217217</td>
      <td>10.217217</td>
      <td>0.204198</td>
      <td>-1.074864</td>
      <td>0.716623</td>
      <td>35.400000</td>
      <td>347.375</td>
      <td>326.207</td>
      <td>347.159354</td>
      <td>134</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 210 columns</p>
</div>



There are over 200 different properties calculated by RDKit, including molecular weight, valence electrons, etc


```python
# I'm going to want these as numpy arrays
X_test_chem  = np.array(pd.DataFrame(X_test_chem))
X_train_chem = np.array(pd.DataFrame(X_train_chem))
X_valid_chem = np.array(pd.DataFrame(X_valid_chem))
```


```python
# Let's see how we do without any more work
regr = GradientBoostingRegressor(
    random_state=1, 
    )
regr.fit(X_train_chem, y_train)

```




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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GradientBoostingRegressor(random_state=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GradientBoostingRegressor<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html">?<span>Documentation for GradientBoostingRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GradientBoostingRegressor(random_state=1)</pre></div> </div></div></div></div>




```python
training_score, validation_score = regr.score(X_train_chem, y_train), regr.score(X_valid_chem, y_valid)
print(f'Training score: {training_score:.2f}, Validation score: {validation_score:.2f}')
```

    Training score: 0.84, Validation score: 0.38


Not a bad start, but it can be better. Let's add preprocessing to the chemical properties dataset


```python
import matplotlib.pyplot as plt
plt.hist(y_train)
plt.show()
```


    
![png](/images/PPBR/output_17_0.png)
    



```python
plt.hist(X_train_chem[155:160,:])
plt.show()
```


    
![png](/images/PPBR/output_18_0.png)
    


The target data is heavily skewed to the high values, the input data is almost always zero for a given dimension but occasionally there will be a non-zero value for some molecule.


```python
x_scaler = QuantileTransformer(
                random_state=1,
                output_distribution='normal',
)
y_scaler = QuantileTransformer(
                random_state=1,
                output_distribution='normal',
) 
```


```python
X_train_scaled = x_scaler.fit_transform(X_train_chem)
X_validation_scaled = x_scaler.transform(X_valid_chem)
y_train_scaled = y_scaler.fit_transform(y_train)
y_validation_scaled = y_scaler.transform(y_valid)

regr = GradientBoostingRegressor(
    random_state=1, 
    )
regr.fit(X_train_scaled, y_train_scaled)
training_score, validation_score = regr.score(X_train_scaled, y_train_scaled), regr.score(X_validation_scaled, y_validation_scaled)
print(f'Training score: {training_score:.2f}, Validation score: {validation_score:.2f}')
```

    Training score: 0.81, Validation score: 0.47



```python
plt.hist(y_train_scaled)
plt.show()
```


    
![png](/images/PPBR/output_22_0.png)
    



```python
plt.hist(X_train_scaled[155:160,:])
plt.show()
```


    
![png](/images/PPBR/output_23_0.png)
    


Transforming the data into a normal distribution helps train the model

### Morgan fingerprinting 
Another type of data to use is a more direct comparison between molecules: molecular fingerprints
These look at atoms near to one another within the molecule. These fingerprints can be used to compare molecules to one another in the dataset and judge similarity


```python
radius = 7 # the default radius
training['mol'] = training['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in training['mol']]

valid['mol'] = valid['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
valid_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in valid['mol']]

test['mol'] = test['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in test['mol']]
```


```python
# from the fingerprints, make similarity matrices 
num_molecules = len(fingerprints)
similarity_matrix = np.zeros((num_molecules, num_molecules))
num_valid_molecules = len(valid_fingerprints)
num_test_molecules= len(test_fingerprints)
```


```python
num_molecules = len(fingerprints)
num_valid_molecules = len(valid_fingerprints)
num_test_molecules= len(test_fingerprints)

similarity_matrix = np.zeros((num_molecules, num_molecules))
test_similarity_matrix = np.zeros((num_test_molecules, num_molecules))
valid_similarity_matrix = np.zeros((num_valid_molecules, num_molecules))

for i in range(num_molecules):
    for j in range(i, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Symmetric matrix

for i in range(num_test_molecules):
    for j in range(0, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(test_fingerprints[i], fingerprints[j])
        test_similarity_matrix[i, j] = similarity

for i in range(num_test_molecules):
    for j in range(0, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(test_fingerprints[i], fingerprints[j])
        test_similarity_matrix[i, j] = similarity

for i in range(num_valid_molecules):
    for j in range(0, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(valid_fingerprints[i], fingerprints[j])
        valid_similarity_matrix[i, j] = similarity
```


```python
similarity_matrix[0:5, 0:10]
```




    array([[1.        , 0.08379888, 0.08547009, 0.09852217, 0.07537688,
            0.11363636, 0.08606557, 0.10144928, 0.08      , 0.07614213],
           [0.08379888, 1.        , 0.05641026, 0.08024691, 0.05732484,
            0.10674157, 0.05853659, 0.06508876, 0.09550562, 0.08609272],
           [0.08547009, 0.05641026, 1.        , 0.07798165, 0.09178744,
            0.08898305, 0.1888412 , 0.08108108, 0.1038961 , 0.06666667],
           [0.09852217, 0.08024691, 0.07798165, 1.        , 0.07142857,
            0.09178744, 0.07423581, 0.10582011, 0.12562814, 0.08426966],
           [0.07537688, 0.05732484, 0.09178744, 0.07142857, 1.        ,
            0.04326923, 0.05803571, 0.05820106, 0.075     , 0.05142857]])




```python
valid_similarity_matrix[0:5, 0:10]
```




    array([[0.08982036, 0.09836066, 0.0718232 , 0.06535948, 0.06944444,
            0.06936416, 0.05670103, 0.05660377, 0.08284024, 0.08571429],
           [0.07352941, 0.0621118 , 0.04524887, 0.07526882, 0.06703911,
            0.04225352, 0.06140351, 0.06770833, 0.05769231, 0.0441989 ],
           [0.08097166, 0.07881773, 0.07307692, 0.05531915, 0.04824561,
            0.07142857, 0.07407407, 0.0720339 , 0.08943089, 0.08219178],
           [0.06756757, 0.05      , 0.07327586, 0.09      , 0.07731959,
            0.09589041, 0.08786611, 0.05188679, 0.08675799, 0.078125  ],
           [0.09876543, 0.0631068 , 0.05681818, 0.09251101, 0.05752212,
            0.09311741, 0.04693141, 0.10480349, 0.08064516, 0.06756757]])



Let's train the model on just these similarity matrices


```python
X_train, X_valid, X_test = similarity_matrix, valid_similarity_matrix, test_similarity_matrix

regr = GradientBoostingRegressor(
    random_state=1, 
    )
regr.fit(X_train, y_train)
f"Training score: {training_score:.2f}, Validation score: {validation_score:.2f}"
```




    'Training score: 0.81, Validation score: 0.47'



It's not better than the chemical property model, but it's almost guaranteed to contain different information. Let's combine them and see if the model is further improved.


```python
# This is just an intermediate check, I'm going to skip the pre-processing.
#So I want to compare the scores with cell 25: Training score: 0.84 Validation score: 0.38

#concatenate chemical data with fingerprint data
X_train = np.concatenate((similarity_matrix, X_train_chem), axis=1)
X_valid = np.concatenate((valid_similarity_matrix, X_valid_chem), axis=1)

regr = GradientBoostingRegressor(
    random_state=1, 
    )
regr.fit(X_train, y_train)
print(f'Training score: {training_score:.2f}, Validation score: {validation_score:.2f}')
```

    Training score: 0.81, Validation score: 0.47


### Determine the right radius to use 
That actually made the model worse at predicting the validation dataset. I'm going to adjust how the Morgan Fingerprints are calculated to see if there's a better way.


```python
num_molecules = len(fingerprints)
num_valid_molecules = len(valid_fingerprints)
num_test_molecules= len(test_fingerprints)

similarity_matrix = np.zeros((num_molecules, num_molecules))
test_similarity_matrix = np.zeros((num_test_molecules, num_molecules))
valid_similarity_matrix = np.zeros((num_valid_molecules, num_molecules))

for radius in range(10):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in training['mol']]
    #test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in test['mol']]
    valid_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in valid['mol']]
    
    for i in range(num_molecules):
        for j in range(i, num_molecules):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    for i in range(num_test_molecules):
        for j in range(0, num_molecules):
            similarity = DataStructs.TanimotoSimilarity(test_fingerprints[i], fingerprints[j])
            test_similarity_matrix[i, j] = similarity
    
    for i in range(num_test_molecules):
        for j in range(0, num_molecules):
            similarity = DataStructs.TanimotoSimilarity(test_fingerprints[i], fingerprints[j])
            test_similarity_matrix[i, j] = similarity
    
    for i in range(num_valid_molecules):
        for j in range(0, num_molecules):
            similarity = DataStructs.TanimotoSimilarity(valid_fingerprints[i], fingerprints[j])
            valid_similarity_matrix[i, j] = similarity

    X_train, X_valid, X_test = similarity_matrix, valid_similarity_matrix, test_similarity_matrix
   
    regr = GradientBoostingRegressor(
    random_state=1, 
    )
    regr.fit(X_train, y_train)
    print("radius: "+str(radius), regr.score(X_train, y_train), regr.score(X_valid, y_valid))
```

    radius: 0 0.5911407088782001 -0.19957336525763902
    radius: 1 0.7307129311812188 0.21432168957971698
    radius: 2 0.7601903491354445 0.09446409128208133
    radius: 3 0.7709970603513945 0.16536085141618106
    radius: 4 0.7822654750730927 0.19376403642887308
    radius: 5 0.799313228929121 0.09288487963697722
    radius: 6 0.7830749173722934 0.07143858347131138
    radius: 7 0.8083843165268636 0.15329793225288801
    radius: 8 0.7991026185815263 0.10111769580646157
    radius: 9 0.7923410240659565 -0.003452249174486388


Looks like a radius of 1 is the right range for this dataset. So we're looking at the central atom and it's most immediate neighbors. 

# Put it all together, train the model
I want to bring the chemical descriptions, the molecular fingerprints, and the preprocessing together


```python
## Get the data 
training, test, valid = split['train'], split['test'], split['valid']
test['mol'] = test['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
training['mol'] = training['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
valid['mol'] = valid['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
y_train, y_valid, y_test = training['Y'].to_frame(), valid['Y'].to_frame(), test['Y'].to_frame()

## Add Chemical properties
X_test_chem = np.array(pd.DataFrame([Descriptors.CalcMolDescriptors(x) for x in test['mol']]))
X_train_chem =np.array(pd.DataFrame([Descriptors.CalcMolDescriptors(x) for x in training['mol']]))
X_valid_chem =np.array(pd.DataFrame([Descriptors.CalcMolDescriptors(x) for x in valid['mol']]))

## Add Morgan fingerprints
radius = 1
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in training['mol']]
test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in test['mol']]
valid_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in valid['mol']]

num_molecules = len(fingerprints)
num_valid_molecules = len(valid_fingerprints)
num_test_molecules= len(test_fingerprints)

similarity_matrix = np.zeros((num_molecules, num_molecules))
test_similarity_matrix = np.zeros((num_test_molecules, num_molecules))
valid_similarity_matrix = np.zeros((num_valid_molecules, num_molecules))



for i in range(num_molecules):
    for j in range(i, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Symmetric matrix

for i in range(num_test_molecules):
    for j in range(0, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(test_fingerprints[i], fingerprints[j])
        test_similarity_matrix[i, j] = similarity

for i in range(num_valid_molecules):
    for j in range(0, num_molecules):
        similarity = DataStructs.TanimotoSimilarity(valid_fingerprints[i], fingerprints[j])
        valid_similarity_matrix[i, j] = similarity
       
X_train = np.concatenate((similarity_matrix, X_train_chem), axis=1)
X_valid = np.concatenate((valid_similarity_matrix, X_valid_chem), axis=1)
X_test = np.concatenate((test_similarity_matrix, X_test_chem), axis=1)

### Preprocessing 
x_scaler = QuantileTransformer(
                random_state=1,
                output_distribution='normal',
)
y_scaler = QuantileTransformer(
                random_state=1,
                output_distribution='normal',
) 

X_train_scaled = x_scaler.fit_transform(X_train) #Dont' cheat, fit the transform on the training data
X_validation_scaled = x_scaler.transform(X_valid)
y_train_scaled = y_scaler.fit_transform(y_train) #Dont' cheat, fit the transform on the training data
y_validation_scaled = y_scaler.transform(y_valid)

### Train and evaulate the model 
regr = GradientBoostingRegressor(
random_state=1, 
)
regr.fit(X_train_scaled, y_train_scaled)
print(f'Training score: {training_score:.2f}, Validation score: {validation_score:.2f}')
```

    Training score: 0.81, Validation score: 0.47


Comapred to where we started (validation score = 0.38) this is a moderate improvement. Let's submit!

# Using the submission framework

Copying submission method from [MapLight](https://github.com/maplightrx/MapLight-TDC/blob/main/submission.ipynb)


```python
benchmark_config = {
    'ppbr_az': ('regression', False),
}
```


```python
group = admet_group(path = 'data/')

for admet_benchmark in [list(benchmark_config.keys())[0]]:
    predictions_list = []
    for seed in tqdm([1, 2, 3, 4, 5]):  
        benchmark = group.get(admet_benchmark)
        predictions = {}
        name = benchmark['name']
        training, test = benchmark['train_val'], benchmark['test']
        test['mol'] = test['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
        training['mol'] = training['Drug'].apply(lambda x: Chem.MolFromSmiles(x)) 
    
        radius=1
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in training['mol']]
        test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048) for mol in test['mol']]
            
        num_molecules = len(fingerprints)
        num_test_molecules = len(test_fingerprints)
        similarity_matrix = np.zeros((num_molecules, num_molecules))
        test_similarity_matrix = np.zeros((num_test_molecules, num_molecules))
    
        for i in range(num_molecules):
            for j in range(i, num_molecules):
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric matrix
        
        for i in range(num_test_molecules):
            for j in range(0, num_molecules):
                similarity = DataStructs.TanimotoSimilarity(test_fingerprints[i], fingerprints[j])
                test_similarity_matrix[i, j] = similarity
    
        X_train, X_test = similarity_matrix, test_similarity_matrix
    
        X_train_chem = np.array(pd.DataFrame([Descriptors.CalcMolDescriptors(x) for x in training['mol']]))
        X_test_chem = np.array(pd.DataFrame([Descriptors.CalcMolDescriptors(x) for x in test['mol']]))
        
        X_train = np.concatenate((X_train, X_train_chem), axis=1)
        X_test = np.concatenate((X_test, X_test_chem), axis=1)

        qt = QuantileTransformer(
            random_state=seed,
            output_distribution='normal',
                    )
        y_train, y_test = training['Y'].to_frame(), test['Y'].to_frame()
        y_train = qt.fit_transform(y_train)
        y_test = qt.transform(y_test)
        
        scaler = QuantileTransformer(
                random_state=seed,
                output_distribution='normal',

                )
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        regr = GradientBoostingRegressor(
            random_state=seed, 
            )
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test).reshape(-1,1)
        y_pred_test = qt.inverse_transform(y_pred).reshape(-1)

    # --------------------------------------------- #
        prediction_dict = {name: y_pred_test}
        predictions_list.append(prediction_dict)
    results = group.evaluate_many(predictions_list)
    print('\n\n{}'.format(results))
```

    Found local copy...
    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [10:34<00:00, 126.99s/it]

    
    
    {'ppbr_az': [7.441, 0.024]}


    


As of 2024/07/09, this is a first place score on the [leaderboard](https://tdcommons.ai/benchmark/admet_group/08ppbr/)!

The current leader is: MapLight + GNN, Jim Notwell, 7.526 ± 0.106


```python

```
