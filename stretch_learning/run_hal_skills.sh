#!/usr/bin/env bash

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

echo $skill_name
echo $model_type
echo $train_type

roslaunch stretch_learning hal_skills.launch \
    skill_name:="$skill_name" \
    model_type:="$model_type" \
    train_type:="$train_type"