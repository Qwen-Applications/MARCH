#!/bin/bash

mkdir -p $WORKSPACE

if [ ! -d "${WORKSPACE}/verl" ]; then
    cp /root/code/verl.zip $WORKSPACE/verl.zip
    unzip -o $WORKSPACE/verl.zip -d $WORKSPACE
fi

if [ ! -d "${WORKSPACE}/MARCH" ]; then
    cp /root/code/MARCH.zip $WORKSPACE/MARCH.zip
    unzip -o $WORKSPACE/MARCH.zip -d $WORKSPACE
fi

# pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip3 config set install.trusted-host mirrors.aliyun.com

# pip3 config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://mirrors.cloud.tencent.com/pypi/simple/ https://pypi.mirrors.ustc.edu.cn/simple/"
# pip3 config set global.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com mirrors.cloud.tencent.com pypi.mirrors.ustc.edu.cn"

pip3 install -r $WORKSPACE/MARCH/requirements/requirements-base.txt

apt install -y rclone


