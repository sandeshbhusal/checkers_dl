#!/bin/bash
while true
do
    # Then we'll have the data.csv file. Run trainer.
    ./trainer --batch-size 100 --num-epochs 10 --testnet testnet --data-file "data.csv"

    # Remove old training data.
    mv data.csv "data-$(date).csv" || true
    rm log.txt || true

    # Then, run against mlpplayer

    echo "total losses as both players"
    for i in {0..5}; do ./checkers mlpplayer mlpplayer 1 2>>log.txt; done
    # Introduce some randomness
    for i in {0..5}; do ./checkers mlpplayer computer 1 2>>log.txt; done
    for i in {0..5}; do ./checkers computer mlpplayer 1 2>>log.txt; done
    # End of randomness
    for i in {0..5}; do ./checkers mlpplayer mlpplayer 1 2>>log.txt; done
    echo $(cat log.txt | grep 'has lost' | sort | uniq -c) >> logbook.txt
    
    killall computer
    killall java
    killall mlpplayer

    # We have a new data.csv now. Also we have a new log.txt
    # Print out the contents of log.txt "grepped" and push to a file.

done
