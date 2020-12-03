# TextSummarization


### Environement Setup
```
$ conda create -n new_env --file req.txt
$ conda activate new_env
```
### Run Extractive Summarization
```
$ python extractive.py -file_path FILE_PATH -top_n N_SENTENCES
```
* `FILE_PATH` is the path of the text file (.txt), `N_SENTENCES` is the number of extracted sentences to be printed.
