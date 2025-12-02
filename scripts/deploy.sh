#!/bin/bash

set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./deploy.sh <version>"
    exit 1
fi

echo "Deploying version $VERSION"

# 更新版本号
python scripts/update_version.py $VERSION

# 提交更改
git add .
git commit -m "Bump version to $VERSION"
git tag -a "v$VERSION" -m "Release version $VERSION"

# 推送到远程
git push origin main
git push origin "v$VERSION"

echo "Deployment triggered for version $VERSION"
