#!/bin/sh
cd /home/claude/fddpg
cat jobs_tune.txt | xargs -P 2 -L 1 sh -c 'python3 train_cfg.py $0 $1 $2 $3 $4 $5 $6 $7 > logs/tune_$0_$1_$5_$6_$7_$2.log 2>&1'
python3 select_cfg.py > logs/select.log 2>&1
cat jobs_final.txt | xargs -P 2 -L 1 sh -c 'python3 train_cfg.py $0 $1 $2 $3 $4 $5 $6 $7 > logs/final_$0_$1_$2.log 2>&1'
echo DONE > logs/pipeline_done
