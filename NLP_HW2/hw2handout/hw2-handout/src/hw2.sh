#!/bin/bash
# Sample script file to run your code. Feel free to change it.
# Run this script using ./hw2.sh train_file test_file
# Example:  ./hw2.sh ./naivebayes.py ../dev_text.txt ../dev_label.txt ../heldout_text.txt ../heldout_pred_nb.txt

echo "Running using train file at" $1 "and the label at" $2 "and test file at" $3 "and save pred at" $4
# python3 naivebayes.py dev_text.txt dev_label.txt heldout_text.txt heldout_pred_nb.txt
python3 $1 $2 $3 $4
