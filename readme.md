# ODSR (Unified)

## Data

Make sure the dataset path matches `common.py::_resolve_dataset_path()`.

Recommended layout:
- ./dataset/douban-book/
- ./dataset/epinions/
- ./dataset/yelp2018/

Each dataset folder should contain:
- train.txt
- test.txt
- trust.txt  (or links.txt)



## Checkpoints

Default behavior is evaluation from checkpoint (no training).
The default checkpoint file prefers capitalized names:

- douban-book -> checkpoints/Douban-Book.pth
- epinions    -> checkpoints/Epinions.pth
- yelp2018    -> checkpoints/Yelp2018.pth

You can override with `--ckpt path/to/file.pth`.


## Run (default: eval from ckpt)

Douban-Book:
- python main.py --dataset douban-book

Epinions:
- python main.py --dataset epinions

Yelp2018:
- python main.py --dataset yelp2018


## Train (overwrite checkpoint)

Add `--train` to retrain and overwrite the default checkpoint for that dataset:

- python main.py --dataset douban-book --train
- python main.py --dataset epinions --train
- python main.py --dataset yelp2018 --train

Notes:
- training uses early stopping with evaluation every `eval_interval`
- checkpoint saved is the best monitored metric (default: recall@20)


## Output style

Terminal output is plain text only:
- no emojis
- no symbolized decorations
- concise one-line training logs and evaluation logs
