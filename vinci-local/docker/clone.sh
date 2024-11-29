#!/bin/bash

# 固定的多个仓库 URL 和分支名称数组
REPO_URLS=("https://gitlab.pjlab.org.cn/cloud/vinci-be.git" "https://gitlab.pjlab.org.cn/openxlabs/vinci-fe.git" "https://gitlab.pjlab.org.cn/cloud/vinci-inference.git" "https://gitlab.pjlab.org.cn/liujinyao/vinci-retrieval.git")
BRANCH_NAMES=("privatization" "privatization" "privatization" "privatization")

# 检查数组长度是否匹配
if [ ${#REPO_URLS[@]} -ne ${#BRANCH_NAMES[@]} ]; then
    echo "Error: The number of repositories and branches must match."
    exit 1
fi

# 循环处理每个仓库
for i in "${!REPO_URLS[@]}"; do
    REPO_URL="${REPO_URLS[$i]}"
    BRANCH_NAME="${BRANCH_NAMES[$i]}"

    # 从仓库 URL 获取目录名
    REPO_DIR=$(basename "$REPO_URL" .git)

    echo "Processing repository $REPO_URL on branch $BRANCH_NAME..."

    # 检查仓库目录是否存在
    if [ ! -d "$REPO_DIR" ]; then
        # 如果目录不存在，克隆仓库
        echo "Cloning repository from $REPO_URL..."
        git clone "$REPO_URL"
        if [ $? -ne 0 ]; then
            echo "Failed to clone repository $REPO_URL."
            continue
        fi
    fi

    # 进入仓库目录
    cd "$REPO_DIR" || exit

    # 拉取最新的远程分支
    echo "Fetching latest changes from remote..."
    git fetch origin

    # 检查指定的分支是否存在
    if git show-ref --verify --quiet refs/heads/"$BRANCH_NAME"; then
        # 如果本地已经有该分支，切换到分支并拉取最新的代码
        echo "Switching to branch $BRANCH_NAME..."
        git checkout "$BRANCH_NAME"
    else
        # 如果本地没有该分支，尝试从远程创建并切换
        echo "Branch $BRANCH_NAME does not exist locally. Checking out from origin..."
        git checkout -b "$BRANCH_NAME" origin/"$BRANCH_NAME"
    fi

    # 拉取最新代码
    echo "Pulling latest changes for branch $BRANCH_NAME..."
    git pull origin "$BRANCH_NAME"

    echo "Repository $REPO_URL is now on branch $BRANCH_NAME."

    # 回到上一级目录以继续处理下一个仓库
    cd ..

done

echo "All repositories processed."
