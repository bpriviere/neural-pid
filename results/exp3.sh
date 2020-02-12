ulimit -n 2048
rm -rf singleintegrator/exp3Empty*
rm -rf singleintegrator/exp3Barrier*
python3 singleintegrator/exp3.py --train
python3 singleintegrator/exp3.py --sim
python3 singleintegrator/exp3.py --plot
