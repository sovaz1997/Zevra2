for i in {1..10}; do
  start=$(( (i - 1) * 1000000 + 1 ));
  end=$(( i * 1000000 ));
  echo "Run generation: $start - $end";
  ./zevra --generate-dataset $start $end ${i}.csv > ${i}.txt &
done
