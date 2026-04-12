#!/bin/bash

function verify_ray_nodes() {
    local expected_nodes=$NNODES
    local max_retries=30
    local retry_interval=10

    echo "Verifying Ray cluster node count (expecting ${expected_nodes} nodes)..."

    for i in $(seq 1 $max_retries); do
        # Get number of nodes from Ray status by counting lines that start with space and "node_"
        local node_count=$(ray status 2>/dev/null | grep -E '^\s+[0-9]+\s+node_' | wc -l)

        if [ "$node_count" -eq "$expected_nodes" ]; then
            echo "Ray cluster node verification successful: ${node_count}/${expected_nodes} nodes"
            return 0
        fi

        if [ $i -eq $max_retries ]; then
            echo "Node verification failed: found ${node_count}/${expected_nodes} nodes"
            return 1
        fi

        echo "Waiting for all nodes to be registered... (${node_count}/${expected_nodes}) attempt ${i}/${max_retries}"
        sleep $retry_interval
    done
}

function init_ray() {
    local max_retries=30
    local retry_interval=10

    if [ $NODE_RANK -eq 0 ]; then
        echo "Starting Ray head node..."
        ray start --head --port=6379 --dashboard-host=0.0.0.0
        echo "Ray head node is ready"
    else
        echo "Connecting to Ray head node..."
        # Add small random delay based on rank to prevent connection storms
#        sleep $(echo "scale=3; $NODE_RANK * 0.5" | bc)
        for i in $(seq 1 $max_retries); do
            ray start --address=${MASTER_ADDR}:6379
            # Wait a moment to check if ray is running
            sleep 5
            if ray status 2>/dev/null; then
                echo "Successfully connected to Ray head node"
                break
            fi
            if [ $i -eq $max_retries ]; then
                echo "Failed to connect to Ray head node"
                exit 1
            fi
            echo "Retrying connection to Ray head node... ($i/$max_retries)"
            sleep $retry_interval
        done
    fi

    # Add verification after initialization
    if ! verify_ray_nodes; then
        echo "Ray cluster node verification failed"
        exit 1
    fi
}
