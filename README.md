# A-STAR

## Load Data
Add your dataset into ```Dataset/```. Put the table data(.csv, .tsv, and so on) in the first level sub-directory, and put the qa pairs data in ```questions/```.
At last, the ```Dataset/```directory should be like this:
```
Dataset/
  Your Dataset/
    table_path/
      table_1.csv
      table_2.csv
      ...
    questions/
      train.json
      dev.json
      test.json
      ...
```

## Run A-STAR
1.Open ASTAR.py and fill in your API Key(If necessary), you can see it in the beginning of file.

2.Install all dependencies in ASTAR.py, like ```ollama```.

3.Run
``` 
python ASTAR.py 
```

4.You can change the backbone model(default model is llama3.1) like:
``` 
python ASTAR.py --engine gpt-4
```
Or you can change the dataset(default dataset is WikiTableQuestions) like:
``` 
python ASTAR.py --dataset TabFact
```
You can set more choices or parameters in ASTAR.py.

## Running Log
You can see the whole process in ```run_log.txt```, and all the candidate answers in ```trial_log.txt```.
