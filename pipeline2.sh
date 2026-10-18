#!/bin/sh
cd /home/claude/fddpg
[ -f logs/pipeline2.lock ] && kill -0 $(cat logs/pipeline2.lock) 2>/dev/null && exit 0
echo $$ > logs/pipeline2.lock
cat jobs_seeds.txt | xargs -P 2 -L 1 sh -c 'python3 train_one.py $0 $1 $2 400 main > logs/main_$0_$1_$2.log 2>&1'
cat jobs_tune2.txt | xargs -P 2 -L 1 sh -c 'python3 train_cfg.py $0 $1 $2 $3 $4 $5 $6 $7 > logs/tune_$0_$1_$5_$6_$7_$2.log 2>&1'
echo DONE > logs/pipeline2_done
