#!/bin/bash

for i in $(seq 0 7)
do
    python run_blip.py --device $i &
done
