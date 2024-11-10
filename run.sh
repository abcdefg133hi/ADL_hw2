mv ./ADL_Homework2/* .
export PYTHONIOENCODING=utf8
python3 hw2.py --model_name "./model" --train_file "./train_0.jsonl" --valid_file ${1} --epochs 0 --batch 4 --beams 10 --output_file ${2}
