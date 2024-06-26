---
title: "Predicting peptide-protein binding with a random forest"
excerpt: "General protein-peptide affinity predicted from biochemical features.  <br/><img src='/images/protein-peptide/thumbnail.PNG' width='533' height='300'>"
collection: portfolio
---
## Intro
TDC-2 introduces the Protein-Peptide Binding Affinity prediction task. The predictive, non-generative task is to learn a model estimating a function of a protein, peptide, antigen processing pathway, biological context, and interaction features. It outputs a binding affinity value or binary label indicating strong or weak binding. The binary label can also include additional biomarkers, such as allowing for a positive label if and only if the binding interaction is specific.
To account for additional biomarkers beyond binding affinity value, our task is specified with a binary label.
TDC-2 provides datasets and benchmarks for a generalized protein-peptide binding interaction prediction task and a TCR-Epitope binding interaction prediction task.

X: protein, peptide, antigen processing pathway, biological context, and interaction features

y:  binding affinity value or binary label indicating strong or weak binding
https://rlearsch.github.io/images/protein-peptide/thumbnail.png

1. Find data sources
1. Explore and visualize the data
1. Clean data,
1. Feature engineer
1. Additional data
1. Train models
1. Deploy best model.

## Find data sources
Data is from TDC: https://tdcommons.ai/benchmark/proteinpeptide_group/overview/


```python
from tdc.benchmark_group.protein_peptide_group import ProteinPeptideGroup
group = ProteinPeptideGroup()
train, val = group.get_train_valid_split() # val dataset will be empty. use the train dataset if fine-tuning desired.
test = group.get_test()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
```

    Found local copy...
    Loading...
    Done!


## Explore and visualize the data


```python
X_train.head()
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
      <th>Sequence_ID</th>
      <th>Sequence</th>
      <th>Protein Target_ID</th>
      <th>Protein Target</th>
      <th>Name</th>
      <th>ALC (sequencing confidence)</th>
      <th>KD (nM)</th>
      <th>Reference</th>
      <th>License</th>
      <th>Unnamed: 7</th>
      <th>...</th>
      <th>lower</th>
      <th>expected</th>
      <th>upper</th>
      <th>protein_or_rna_sequence</th>
      <th>gene_type</th>
      <th>X1</th>
      <th>X2</th>
      <th>ID1</th>
      <th>ID2</th>
      <th>KD (nm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2165</th>
      <td>NaN</td>
      <td>FADMPDYAGNVGK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>NaN</td>
      <td>&gt;85</td>
      <td>Putative binder</td>
      <td>DOI:10.26434/chemrxiv-2023-tws4n\n</td>
      <td>Creative Commons Attribution-NonCommercial-NoD...</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>Could not find type of gene</td>
      <td>FADMPDYAGNVGK</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>NaN</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>NaN</td>
      <td>GEEFHDYRDYADK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>NaN</td>
      <td>&gt;85</td>
      <td>Putative binder</td>
      <td>DOI:10.26434/chemrxiv-2023-tws4n\n</td>
      <td>Creative Commons Attribution-NonCommercial-NoD...</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>Could not find type of gene</td>
      <td>GEEFHDYRDYADK</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>NaN</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>215</th>
      <td>NaN</td>
      <td>ASFAEYWNLLSAK</td>
      <td>MDM2</td>
      <td>MDM2</td>
      <td>NaN</td>
      <td>88</td>
      <td>Putative binder</td>
      <td>DOI:10.1038/s42004-022-00737-w</td>
      <td>Creative Commons Attribution 4.0 International...</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKD...</td>
      <td>protein-coding</td>
      <td>ASFAEYWNLLSAK</td>
      <td>MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKD...</td>
      <td>NaN</td>
      <td>MDM2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4156</th>
      <td>NaN</td>
      <td>FLNDKEDYSSAQK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>NaN</td>
      <td>&gt;85</td>
      <td>Putative binder</td>
      <td>DOI:10.26434/chemrxiv-2023-tws4n\n</td>
      <td>Creative Commons Attribution-NonCommercial-NoD...</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>Could not find type of gene</td>
      <td>FLNDKEDYSSAQK</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>NaN</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3630</th>
      <td>NaN</td>
      <td>WMMPMDAKDYAEK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>NaN</td>
      <td>&gt;85</td>
      <td>Putative binder</td>
      <td>DOI:10.26434/chemrxiv-2023-tws4n\n</td>
      <td>Creative Commons Attribution-NonCommercial-NoD...</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>Could not find type of gene</td>
      <td>WMMPMDAKDYAEK</td>
      <td>Protein sequence not found for the given gene ...</td>
      <td>NaN</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



The X data are amino acid sequences, the binding targets they were measured against, and some information about the source of the data.


```python
y_train.value_counts()
```




    Y
    0    440
    1      3
    Name: count, dtype: int64



The y data are binary: bind or doesn't bind. Only 3 of 443 sequence-target pairs result in binding.

## Feature Engineer
I found this package to  extract biochemical features from amino acid sequences, I'm going to try it: 

https://github.com/amckenna41/protpy


```python
import protpy as protpy
import pandas as pd
```


```python
function_list = [protpy.amino_acid_composition, 
                 protpy.dipeptide_composition,
                 protpy.tripeptide_composition,
                 protpy.moreaubroto_autocorrelation,
                 #using default parameters: lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"], normalize=True
                 protpy.moran_autocorrelation,
                 #using default parameters: lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"], normalize=True
                 protpy.geary_autocorrelation,
                 #using default parameters: lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"], normalize=True,
                 protpy.conjoint_triad,
                 #protpy.sequence_order_coupling_number_, ### <--------- this returns a float, i will have to handle it separately 
                 protpy.sequence_order_coupling_number,
                 #using default parameters: lag=30, distance_matrix="schneider-wrede"
                 protpy.quasi_sequence_order,
                 #using default parameters: lag=30, weight=0.1, distance_matrix="schneider-wrede"
                ]

X_train = X_train[['X1','Protein Target_ID']]
X_train = X_train.reset_index()

for function in function_list:
    df_list=[]
    for index, sequence in enumerate(X_train['X1']):
        AA = function(sequence)
        df_list.append(AA)
    tempdf = pd.concat(df_list)
    tempdf=tempdf.reset_index(drop=True)
    X_train = X_train.join(tempdf)
X_train['SOCN'] = X_train['X1'].apply(protpy.sequence_order_coupling_number_)

#There's probably a faster way to do this, but this way works - I had trouble applying protpy functions to entire dataframes in parallel. 
```


```python
X_train.head()
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
      <th>index</th>
      <th>X1</th>
      <th>Protein Target_ID</th>
      <th>A</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>...</th>
      <th>QSO_SW42</th>
      <th>QSO_SW43</th>
      <th>QSO_SW44</th>
      <th>QSO_SW45</th>
      <th>QSO_SW46</th>
      <th>QSO_SW47</th>
      <th>QSO_SW48</th>
      <th>QSO_SW49</th>
      <th>QSO_SW50</th>
      <th>SOCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2165</td>
      <td>FADMPDYAGNVGK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>0.000</td>
      <td>7.692</td>
      <td>15.385</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.913</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1207</td>
      <td>GEEFHDYRDYADK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>23.077</td>
      <td>15.385</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.913</td>
    </tr>
    <tr>
      <th>2</th>
      <td>215</td>
      <td>ASFAEYWNLLSAK</td>
      <td>MDM2</td>
      <td>23.077</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.860</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4156</td>
      <td>FLNDKEDYSSAQK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.024</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3630</td>
      <td>WMMPMDAKDYAEK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>7.692</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.325</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9567 columns</p>
</div>




```python
y_train =  y_train.reset_index(drop=True)
y_train[y_train=='1']
```




    155    1
    391    1
    412    1
    Name: Y, dtype: object




```python
X_train.iloc[[155, 391, 412]]
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
      <th>index</th>
      <th>X1</th>
      <th>Protein Target_ID</th>
      <th>A</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>...</th>
      <th>QSO_SW42</th>
      <th>QSO_SW43</th>
      <th>QSO_SW44</th>
      <th>QSO_SW45</th>
      <th>QSO_SW46</th>
      <th>QSO_SW47</th>
      <th>QSO_SW48</th>
      <th>QSO_SW49</th>
      <th>QSO_SW50</th>
      <th>SOCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>155</th>
      <td>17</td>
      <td>TSFAAYWAALSAK</td>
      <td>MDM2</td>
      <td>38.462</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.641</td>
    </tr>
    <tr>
      <th>391</th>
      <td>1</td>
      <td>ASFAAYWNLLSP</td>
      <td>MDM2</td>
      <td>25.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>8.333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.009</td>
    </tr>
    <tr>
      <th>412</th>
      <td>7</td>
      <td>TSFAEYANLLAP</td>
      <td>MDM2</td>
      <td>25.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.333</td>
      <td>8.333</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.073</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9567 columns</p>
</div>



All 3 positive data points are with Protein Target MDM2. I'm going to only work with the subset of data focused on MDM2 as a down-sampling technique to remove negative signal.


```python
MDM2_list = X_train.index[X_train['Protein Target_ID']=='MDM2'].tolist()
X_train_MDM2 = X_train[X_train['Protein Target_ID']=='MDM2']
y_trian_MDM2 = y_train.iloc[MDM2_list]
#Also drop the Protein Target ID column because... we don't need it anymore.
X_train_MDM2 = X_train_MDM2.drop(columns=['index','X1','Protein Target_ID'])
X_train_MDM2.head()
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
      <th>A</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>K</th>
      <th>L</th>
      <th>...</th>
      <th>QSO_SW42</th>
      <th>QSO_SW43</th>
      <th>QSO_SW44</th>
      <th>QSO_SW45</th>
      <th>QSO_SW46</th>
      <th>QSO_SW47</th>
      <th>QSO_SW48</th>
      <th>QSO_SW49</th>
      <th>QSO_SW50</th>
      <th>SOCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>23.077</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>15.385</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.860</td>
    </tr>
    <tr>
      <th>15</th>
      <td>38.462</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.662</td>
    </tr>
    <tr>
      <th>24</th>
      <td>53.846</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>15.385</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.815</td>
    </tr>
    <tr>
      <th>49</th>
      <td>7.692</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.957</td>
    </tr>
    <tr>
      <th>57</th>
      <td>38.462</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.586</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9564 columns</p>
</div>



### Adding SMOTE oversampling
I want to do some strategic over-sampling as well to boost up the positive signals. 


```python
import imblearn.over_sampling
sm = imblearn.over_sampling.SMOTE(random_state=42, k_neighbors=2)
X_res, y_res = sm.fit_resample(X_train_MDM2, y_trian_MDM2)
X_res
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
      <th>A</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>I</th>
      <th>K</th>
      <th>L</th>
      <th>...</th>
      <th>QSO_SW42</th>
      <th>QSO_SW43</th>
      <th>QSO_SW44</th>
      <th>QSO_SW45</th>
      <th>QSO_SW46</th>
      <th>QSO_SW47</th>
      <th>QSO_SW48</th>
      <th>QSO_SW49</th>
      <th>QSO_SW50</th>
      <th>SOCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.077000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>15.385000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.860000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.462000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.662000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>53.846000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7.692000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>15.385000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.815000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.692000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.957000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38.462000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.692000</td>
      <td>7.692000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.586000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>37.999062</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.286560</td>
      <td>7.714043</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>7.427483</td>
      <td>8.000637</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.655856</td>
    </tr>
    <tr>
      <th>58</th>
      <td>25.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.755633</td>
      <td>8.333000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>16.667000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.105483</td>
    </tr>
    <tr>
      <th>59</th>
      <td>28.483696</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.176586</td>
      <td>8.167122</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1.990536</td>
      <td>14.344450</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.961207</td>
    </tr>
    <tr>
      <th>60</th>
      <td>25.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.520798</td>
      <td>8.333000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>16.667000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.713924</td>
    </tr>
    <tr>
      <th>61</th>
      <td>34.265745</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>7.891807</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>5.294318</td>
      <td>10.489607</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.443999</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 9564 columns</p>
</div>



## Train the model
Random forest has worked best for me


```python
import numpy as np
#from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```


```python
rf_classifier_smote = RandomForestClassifier(random_state=42)
#X_train, y_train = X_res, y_res
rf_classifier_smote.fit(X_res, y_res)
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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;RandomForestClassifier<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(random_state=42)</pre></div> </div></div></div></div>



Process the test data in the same way as the training data


```python
#function_list = [protpy.amino_acid_composition, 
#                 protpy.dipeptide_composition,
#                 protpy.tripeptide_composition,
#                 protpy.moreaubroto_autocorrelation,
#                 #using default parameters: lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"], normalize=True
#                 protpy.moran_autocorrelation,
#                 #using default parameters: lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"], normalize=True
#                 protpy.geary_autocorrelation,
#                 #using default parameters: lag=30, properties=["CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102", "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201"], normalize=True,
#                 protpy.conjoint_triad,
#                 #protpy.sequence_order_coupling_number_, ### <--------- this returns a float, i will have to handle it separately 
#                 protpy.sequence_order_coupling_number,
#                 #using default parameters: lag=30, distance_matrix="schneider-wrede"
#                 protpy.quasi_sequence_order,
#                 #using default parameters: lag=30, weight=0.1, distance_matrix="schneider-wrede"
#                ]
#
X_test = X_test[['X1','Protein Target_ID']]
X_test = X_test.reset_index()

for function in function_list:
    df_list=[]
    for index, sequence in enumerate(X_test['X1']):
        AA = function(sequence)
        df_list.append(AA)
    tempdf = pd.concat(df_list)
    tempdf=tempdf.reset_index(drop=True)
    X_test = X_test.join(tempdf)
X_test['SOCN'] = X_test['X1'].apply(protpy.sequence_order_coupling_number_)

#There's probably a faster way to do this, but this way works - I had trouble applying protpy functions to entire dataframes in parallel. 
```


```python
X_test.head()
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
      <th>index</th>
      <th>X1</th>
      <th>Protein Target_ID</th>
      <th>A</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
      <th>H</th>
      <th>...</th>
      <th>QSO_SW42</th>
      <th>QSO_SW43</th>
      <th>QSO_SW44</th>
      <th>QSO_SW45</th>
      <th>QSO_SW46</th>
      <th>QSO_SW47</th>
      <th>QSO_SW48</th>
      <th>QSO_SW49</th>
      <th>QSO_SW50</th>
      <th>SOCN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3910</td>
      <td>LDLKDYADQRKAK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>23.077</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.133</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4145</td>
      <td>TQHLDKHDYAVYK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4277</td>
      <td>SKFMFDPRDYAAK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>0.000</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.074</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3523</td>
      <td>EFVHDLKDYAAPK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>7.692</td>
      <td>7.692</td>
      <td>0.0</td>
      <td>7.692</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.177</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3548</td>
      <td>PAFFWDLNDYAFK</td>
      <td>Anti-HA antibody 12ca5</td>
      <td>15.385</td>
      <td>0.0</td>
      <td>15.385</td>
      <td>0.000</td>
      <td>23.077</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.147</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 9567 columns</p>
</div>




```python
X_test = X_test.drop(columns=['index','X1','Protein Target_ID'])
```


```python
predictions = rf_classifier_smote.predict(X_test)
## The evaluate function requires a very particular structure for the data ##
predictions = predictions.astype(str)
predictions = predictions.astype(object)
out = group.evaluate(predictions)
out
```




    [0.75, 0.1935483870967742, 0.993231386312359, 0.3076923076923077]



The F-score (0.31) is not great, but may be acceptable for such an imbalanced dataset. Based on my knowledge of the biological problem, it seems that you'd rather have false-positives than false-negatives. In other words, it's okay to sacrifice some precision for improved recall, which is not captured in F score.

Another approach that may be useful is reducing the dimension of the sequence by using the commonly used amino acid residue functional groups: 
1. Cationic side chains
1. Anionic side chains
1. Polar uncharged side chains
1. Hydrophobic side chains
1. Special cases

As seen here: https://www.technologynetworks.com/applied-sciences/articles/essential-amino-acids-chart-abbreviations-and-structure-324357 
