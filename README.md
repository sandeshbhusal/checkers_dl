# How to use this thing?

Run `make`. It is going to take ~2 mins.

Then, run a bunch of games:

`for i in {0..100}; do ../checkers ../computer ../computer 1; done`. The checkers server will automatically dump gameplays in the current directory inside `data.csv`.

Then, run the trainer.

`./trainer --batch-size 2000 --num-epochs 10 --testnet testnet --data-file data.csv`

Then you can run the game.
