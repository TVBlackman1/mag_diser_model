#!/bin/bash

# save results to new directory if training finally correctly

SOURCE_DIR="results"
DEST_DIR="results_resolved"

mkdir -p "$DEST_DIR"

# Ищем только директории первого уровня в results
find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    name=$(basename "$dir")
    # Пропускаем, если уже скопировано
    if [ -d "$DEST_DIR/$name" ]; then
        continue
    fi
    # Проверяем наличие нужного файла
    if [ -f "$dir/critic_loss_plot.png" ]; then
        cp -r "$dir" "$DEST_DIR/"
    fi
done