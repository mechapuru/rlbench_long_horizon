#!/bin/bash

DEMO_FILE="examples/demo_custom.py"
TASKS_DIR="rlbench/tasks"

# Get the original task from the demo file to revert back to it at the end.
ORIGINAL_TASK=$(grep -oP '(?<=from rlbench.tasks import )[A-Z][a-zA-Z0-9_]*' "$DEMO_FILE")
if [ -z "$ORIGINAL_TASK" ]; then
    echo "Could not find the original task in $DEMO_FILE."
    exit 1
fi

# Find all task files, excluding __init__.py
for task_file in $(ls $TASKS_DIR/*.py | grep -v '__init__.py'); do
    # Extract the task name from the file path (e.g., /path/to/close_box.py -> close_box)
    TASK_SNAKE_CASE=$(basename "$task_file" .py)

    echo "--------------------------------------------------"
    echo "Running task: $TASK_SNAKE_CASE"
    echo "--------------------------------------------------"

    # Convert snake_case to CamelCase for the class name
    TASK_CAMEL_CASE=$(echo "$TASK_SNAKE_CASE" | awk -F_ '{for(i=1;i<=NF;i++) printf "%s", toupper(substr($i,1,1)) substr($i,2); print ""}')

    # Find the current task name in the demo file
    CURRENT_TASK=$(grep -oP '(?<=from rlbench.tasks import )[A-Z][a-zA-Z0-9_]*' "$DEMO_FILE")
    if [ -z "$CURRENT_TASK" ]; then
        echo "Could not find the current task in $DEMO_FILE."
        # Revert to original and exit
        sed -i "s/$TASK_CAMEL_CASE/$ORIGINAL_TASK/g" "$DEMO_FILE"
        exit 1
    fi

    # Replace the old task name with the new one
    sed -i "s/$CURRENT_TASK/$TASK_CAMEL_CASE/g" "$DEMO_FILE"

    # Run the demo script
    python3 "$DEMO_FILE"

done

echo "--------------------------------------------------"
echo "All tasks finished. Reverting to original task: $ORIGINAL_TASK"
echo "--------------------------------------------------"
sed -i "s/$(grep -oP '(?<=from rlbench.tasks import )[A-Z][a-zA-Z0-9_]*' "$DEMO_FILE")/$ORIGINAL_TASK/g" "$DEMO_FILE"

echo "Done."