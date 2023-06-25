# The data required for the WebTables dataset

## Start from the pre-processed data

Please download the required related tables relationship data we extracted from the following [link](https://drive.google.com/drive/folders/1tTZkrLgwxvpBji6HurqsNlvwroNKSZiq).

## Process the required related table data from scratch
1. Download the raw dataset from this [link](https://github.com/megagonlabs/sato/tree/master/table_data) (Multi-only), place in the [raw_data folder](https://github.com/ysunbp/CORDA/tree/main/sato_data/raw_data) (Remember to delete the init files in each of the Ki folder, they are created to set up the directories in GitHub).
2. Run [generate_rel_jsonl.py](https://github.com/ysunbp/CORDA/tree/main/sato_data/generate_rel_jsonl.py) to generate the related table relationship jsonl files.
