# ODSR 

## Data


Recommended layout:
- ./dataset/douban-book/
- ./dataset/epinions/
- ./dataset/yelp2018/


## Checkpoints

Default behavior is evaluation from checkpoint (no training).
The default checkpoint file prefers capitalized names:

- douban-book -> checkpoints/Douban-Book.pth
- epinions    -> checkpoints/Epinions.pth
- yelp2018    -> checkpoints/Yelp2018.pth

You can override with `--ckpt path/to/file.pth`.


## Run

Douban-Book:
- python main.py --dataset douban-book

Epinions:
- python main.py --dataset epinions

Yelp2018:
- python main.py --dataset yelp2018


## Train

Add `--train` to retrain and overwrite the default checkpoint for that dataset:

- python main.py --dataset douban-book --train
- python main.py --dataset epinions --train
- python main.py --dataset yelp2018 --train

Notes:
- training uses early stopping with evaluation every `eval_interval`
- checkpoint saved is the best monitored metric (default: recall@20)



