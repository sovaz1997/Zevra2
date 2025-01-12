for i in {1..10}; do
  start=$(( (i - 1) * 4029484 + 1 ));
  end=$(( i * 4029484 ));
  echo "Run generation: $start - $end";
  ./zevra --generate-dataset $start $end ${i}.csv > ${i}.txt &
done
