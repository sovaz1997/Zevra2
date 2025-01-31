#!/bin/bash

train_data_dir="dataset"
logs_dir="logs"

mkdir -p "$train_data_dir" "$logs_dir"

gamesCount=12
threads=10
gamesPerThread=$((gamesCount / threads))

for ((i=1; i<=threads; i++)); do
  echo "Run generation: $start - $end"

  file_name="$train_data_dir/${i}.csv"
  log_file="$logs_dir/${i}.txt"

  # random number from 0 to 1000000
  seed=$((RANDOM))
  echo "Seed: $seed"

  ./zevra --generate-dataset $gamesPerThread $seed $file_name > log_file &
done

wait
