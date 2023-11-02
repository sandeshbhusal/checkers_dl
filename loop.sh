#!/bin/bash

# This file will run the training + running with the player.

# First of all, remove the data.csv file.
rm data.csv || true

# Then remove the log.
rm log.txt || true

# Play 100 games computer vs computer.
for i in {0..100}; do ./checkers ./computer ./computer 1 2>>log.txt; done

while true
do
    # Then we'll have the data.csv file. Run trainer.
    ./trainer --batch-size 500 --num-epochs 10 --testnet testnet --data-file data.csv

    # Remove old training data.
    rm data.csv || true
    rm log.txt || true

    # Then, run against mlpplayer

    for i in {0..100}; do ./checkers computer mlpplayer 1 2>>log.txt; done

    # We have a new data.csv now. Also we have a new log.txt
    # Print out the contents of log.txt "grepped" and push to a file.

    echo $(cat log.txt | grep 'has lost' | sort | uniq -c) >> logbook.txt
done