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

## Predict
run `scripts/predict.py`
