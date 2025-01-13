#!/bin/bash

# Параметр для подстановки {epoch}
epoch=$1

# Список файлов для переименования
files=(
    "./fc1.{epoch}.weights.csv"
    "./fc2.{epoch}.weights.csv"
    "./fc3.{epoch}.weights.csv"
)

for file in "${files[@]}"; do
    current_file="${file//\{epoch\}/$epoch}"

    new_file="${file/\.\{epoch\}/}"

    if [ -f "$current_file" ]; then
        mv "$current_file" "$new_file"
        echo "Renamed: $current_file -> $new_file"
    else
        echo "File not found: $current_file"
    fi
done
