#!/bin/bash
while true
do
    # Then we'll have the data.csv file. Run trainer.
    ./trainer --batch-size 100 --num-epochs 10 --testnet testnet --data-file "data.csv"

    # Remove old training data.
    rm log.txt || true

    # Then, run against mlpplayer

    for i in {0..2}; do ./checkers ai 'java PlayerHelper' 1 2>>log.txt; done
    for i in {0..2}; do ./checkers 'java PlayerHelper' ai 1 2>>log.txt; done

    # We have a new data.csv now. Also we have a new log.txt
    # Print out the contents of log.txt "grepped" and push to a file.

done
