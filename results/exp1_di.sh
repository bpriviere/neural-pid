ulimit -n 2048
rm -rf doubleintegrator/exp1Empty*
rm -rf doubleintegrator/exp1Barrier*
python3 doubleintegrator/exp1.py --train
python3 doubleintegrator/exp1.py --sim
python3 doubleintegrator/exp1.py --plot