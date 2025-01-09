#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <file_name> [number_of_lines_to_use]"
    exit 1
fi

# Input file
INPUT_FILE="$1"

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "File '$INPUT_FILE' does not exist."
    exit 1
fi

# Determine total lines to use
if [ $# -eq 2 ]; then
    TOTAL_LINES_TO_USE=$2
    FILE_LINES=$(wc -l < "$INPUT_FILE")
    if [ "$TOTAL_LINES_TO_USE" -gt "$FILE_LINES" ]; then
        echo "The file has only $FILE_LINES lines, but you requested $TOTAL_LINES_TO_USE."
        exit 1
    fi
else
    TOTAL_LINES_TO_USE=$(wc -l < "$INPUT_FILE")
fi

# Calculate lines for training (80%) and validation (20%)
TRAIN_LINES=$((TOTAL_LINES_TO_USE * 80 / 100))
VALIDATE_LINES=$((TOTAL_LINES_TO_USE - TRAIN_LINES))

# Output file names
TRAIN_FILE="train_$(basename "$INPUT_FILE")"
VALIDATE_FILE="validate_$(basename "$INPUT_FILE")"

# Split using sed
sed -n "1,${TRAIN_LINES}p" "$INPUT_FILE" > "$TRAIN_FILE"
sed -n "$((TRAIN_LINES + 1)),$((TRAIN_LINES + VALIDATE_LINES))p" "$INPUT_FILE" > "$VALIDATE_FILE"

# Print summary
echo "File split completed:"
echo "Training dataset: $TRAIN_FILE ($TRAIN_LINES lines)"
echo "Validation dataset: $VALIDATE_FILE ($VALIDATE_LINES lines)"
