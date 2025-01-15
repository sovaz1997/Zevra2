#!/bin/bash

train_data_dir="train-data"
logs_dir="logs"

mkdir -p "$train_data_dir" "$logs_dir"

startPoint=0
lastPoint=100000
threads=10
batchSize=$(( (lastPoint - startPoint) / threads ))

for ((i=1; i<=threads; i++)); do
  start=$(( (i - 1) * batchSize + startPoint ))
  end=$(( i * batchSize + startPoint ))
  echo "Run generation: $start - $end"

  output_csv="$train_data_dir/${i}.csv"
  log_file="$logs_dir/${i}.txt"

  ./zevra --generate-dataset "$start" "$end" "$output_csv" > "$log_file" &
done

wait
