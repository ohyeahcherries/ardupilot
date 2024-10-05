#!/bin/bash

for i in $(seq 1 $1); do
	I="$(($i-1))"
	echo "Running instance $I"
	port=$((14551+$I*10))
	gnome-terminal --title="pilot $i" --working-directory=$HOME/Desktop/ArduPilot/ardupilot/ArduPlane -e "/home/user/Desktop/ArduPilot/ardupilot/Tools/autotest/sim_vehicle.py -v ArduPlane --instance=$I --sysid=$i --out=127.0.0.1:$port
" &
done
sleep 10

masters=""
for i in $(seq 1 $1); do
	I="$(($i-1))"
	
	port=$((14550 + $I*10))
	masters="${masters} --master=127.0.0.1:$port"
done

echo $masters

mavproxy.py ${masters} --map --console --load-module horizon  # --load module swarm # 

