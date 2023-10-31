# How to use this thing?

Run `make`. It is going to take ~2 mins.

`mkdir data` to create a new data dir.

Then, `cd data/`

Then, run a bunch of games:

`for i in {0..100}; do ../checkers ../computer ../computer 1`. The checkers server will automatically dump gameplays in the current directory.

`cd ..`

Then, compile the neural net.

`g++ nn.cpp -o trainer -lpthread -I.`

Then, run the trainer.

`./trainer`