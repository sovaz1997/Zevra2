for i in {1..10}; do
  start=$(( (i - 1) * 200000 + 1 ));
  end=$(( i * 200000 ));
  echo "Run generation: $start - $end";
  ./zevra --generate-dataset $start $end ${i}.csv > ${i}.txt &
done
