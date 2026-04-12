#!/bin/bash
# 同步所有节点

local_ip=$( ifconfig eth0 | grep "inet " | awk '{print $2}' )

echo "RANK${RANK} ${local_ip}" > ./common_ip_${RANK}.txt

counter=`cat ./common_ip_*.txt | wc -l `
while [ $counter -lt ${NNODES} ]
do
    echo "Wait for all nodes to be ready, current counter: ${counter}, all node: ${NNODES}"
    sleep 5
    counter=`cat ./common_ip_*.txt | wc -l`
done

sleep 10
rm -rf ./common_ip_*.txt
