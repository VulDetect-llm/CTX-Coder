#!/bin/bash

PROJECT_ROOT="Your Project Root Directory" # replace into your download github project root directory
DOXYGEN_CONFIG="Doxyfile"

initial_dir=$(pwd)

project_dirs=($(ls -d $PROJECT_ROOT/*))

for project_dir in "${project_dirs[@]}"
do
    if [ -d "$project_dir/doxygen" ]; then
        if [ -d "$project_dir/doxygen/xml" ] && find "$project_dir/doxygen/xml" -name "*.xml" | grep -q .; then
            echo "已存在 Doxygen XML 文件，跳过项目：$project_dir"
            continue
        fi
    fi

    cp "$DOXYGEN_CONFIG" "$project_dir/Doxyfile"

    cd "$project_dir"

    echo "正在生成 Doxygen 文档，项目：$project_dir"

    if doxygen Doxyfile; then
        echo "Doxygen 文档生成成功：$project_dir"
    else
        echo "Doxygen 生成失败，项目：$project_dir"
    fi
    
    cd "$initial_dir"
done
