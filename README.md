# general-ml

User guide:
1. ```git clone https://github.com/monolabs/general-ml.git```
2. ```cd general-ml```
3. ```pip install -r requirements.txt```
4. create experiment folder in side the folder "runs"
5. create json file for config (sample provided). E.g. change the column/variable names accordingly. 
6. put data csv in "data" folder
7. ```python run_optimization.py --data_path <csv path> --save_dir <directory path to save runs (without trailing "/")> --config_path <path to json config>```

Example: ```python run_optimization.py --data_path data/titanic.csv --save_dir runs/example_run --config_path runs/example_run/example_config.json```

After optimization is complete, study and best model is saved in the folder specified in (4).