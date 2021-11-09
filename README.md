# TEIM
**TEIM-Res** is a deep learning-based model to predict the **residue-level** TCR-epitope interactions, including the distances and the contact probabilities between all residue pairs from CDR3βs and epitopes. Our model only takes the primary sequences of CDR3βs and the epitopes as input.

## Environment
0. Instal anaconda.
1. Use `teim.yml` to create a new environment `teim` and install requirements (see [anaconda doc](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more information):
    ```
    conda env create -f teim.yml
    ```
2. Install [ANARCI](https://anaconda.org/bioconda/anarci) for CDR3 numbering.
    ```
    conda install -c bioconda anarci
    ```

## Predict the residue-level interactions of TCR-epitope pairs
1. Put your input TCR-epitope sequence pairs in the `inputs` directory. `inputs/samples.csv` is an example file. The TCRs are represented by their CDR3β sequences and the epitopes are represented by their sequences.
2. run 
    ```
    python scripts/predict.py
    ```
3. The predicted distance matrices and contact site matrices are in `outputs`.

**Still in progress**


