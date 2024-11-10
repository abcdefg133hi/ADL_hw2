# ADL Homework2

## Reproduce the Result
```
./download.sh
./run.sh private.jsonl output.jsonl
```

## Training
```
python3 hw2.py model_name "google/mt5-small" --train_file "train.jsonl" --valid_file "public.jsonl" --epochs 5 --beams 10 --output_file "public_result.jsonl"
```

## Prediction
```
python3 hw2.py model_name "google/mt5-small" --train_file "train.jsonl" --valid_file "prediction.jsonl" --epochs 5 --beams 10 --output_file "submission.jsonl"
```

## Plotting
The data for learning curve are stored inside "learn.json". One can type the following command to plot the learning curve:
```
python3 plotLearningCurve.py
```

## Other Info.
The attempt for different hyperparameters in model generations is stored inside the file "results.json". (b: beams, p:top-p, t:temperature and k is always set to 10.) For 1+1+5 (see the reports for advanced meaning), I stored them into "results_pre.json".

## Have Fun ^-^
