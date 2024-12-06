#!/bin/bash
export DB_ID=1
export THREADS=6
export MODULE_ID=`curl -s -k -u "redis@redis.com:redis" https://localhost:9443/v1/bdbs/$DB_ID | jq '.module_list[] | select(.module_name=="search").module_id' | tr -d '"'`

curl -o /dev/null -s -k -u "redis@redis.com:redis" -X PUT https://localhost:9443/v1/bdbs/$DB_ID -H "Content-Type:application/json" -d '{
    "sched_policy": "mnp",
    "conns": 32
}'

sleep 1

curl -o /dev/null -s -k -u "redis@redis.com:redis" https://localhost:9443/v1/bdbs/$DB_ID/modules/upgrade -H "Content-Type:application/json" -d '{
    "modules": [
      {
        "module_name": "search",
        "new_module_args": "MT_MODE MT_MODE_FULL WORKER_THREADS '$THREADS'",
        "current_module": "'$MODULE_ID'",
        "new_module": "'$MODULE_ID'"
      }
    ]
}'
