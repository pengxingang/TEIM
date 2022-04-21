# TEIM: *TCR-Epitope Interaction Modeling*
This repository contains the codes of TEIM (**T**CR-**E**pitope **I**nteraction **M**odeling). TEIM is a deep learning-based model to predict the TCR-epitope interactions, including two submodels TEIM-Res (TEIM at **Res**idue level) and TEIM-Samp (TEIM at **Samp**le level). 

Both models only takes the primary sequences of CDR3βs and the epitopes as input. **TEIM-Res** predicts the distances and the contact probabilities between all residue pairs of CDR3βs and epitopes. **TEIM-Samp** predicts whether the CDR3βs and epitopes can bind to each other.

## Environment
0. Install Anaconda.
1. Use `teim.yml` to create a new environment `teim` and install requirements (see [anaconda doc](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more information):
    ```
    conda env create -f teim.yml
    ```
2. Install [ANARCI](https://anaconda.org/bioconda/anarci) for CDR3 numbering on the new environment.
    ```
    conda activate teim
    conda install -c bioconda anarci
    ```

## Inference
### Predict the residue-level interactions of TCR-epitope pairs
1. Put your input TCR-epitope sequence pairs in the `inputs/inputs.csv` file. The TCRs are represented by their CDR3β sequences and the epitopes are represented by their sequences in the following format:
    | cdr3             | epitope   |
    | ---------------- | --------- |
    | CASAPGLAGGRPEQYF | LLFGYPVYV |
    | CASRGAAGGRPQYF   | MLWGYLQYV |
    | CASRPGLAGGRAEQYF | FTDSSVWA  |

2. Run 
    ```
    python scripts/inference_res.py
    ```
3. The predicted distance matrices and contact site matrices are in the `outputs` directory: 
    - The predicted distance matrix and contact matrix are in the files names as `dist_<cdr3>_<epitope>.csv` and `site_<cdr3>_<epitope>.csv`, respectively. 
    - The rows and columns of the matrices represent the CDR3βs and epitopes, respectively.
    - The values in the distance matrix stand for the distances of residue pairs (unit: angstrom) and the values in the contact matrix stand for the predicted contact scores (probabilities) of residue pairs (range from 0 to 1). 


### Predict the sample-level interactions of TCR-epitope pairs
1. Put your input TCR-epitope sequence pairs in the `inputs/inputs_bd.csv` file. The format is the same as `inputs/inputs.csv` (residue-level input file).
2. Run 
    ```
    python scripts/inference_samp.py
    ```
3. The predicted sample-level binding scores are in the `outputs/sample_level_binding.csv`. The `binding` column in the file represent the predicted sample-level binding scores (probabilities) of the TCR-epitope pair.



