#!/bin/bash
echo Script name: $0
echo $# arguments
if [ $# -ne 4 ]; then
        echo "illegal number of parameters"
        echo "Argument: project_name zone service_account startup_script"
else
	project_name=$1
	zone=$2
	service_account=$3
	startup_script=$4
	#array=( "f1-micro" "n1-standard-1" "n1-standard-2" "n1-standard-4" "n1-standard-8" )
	array=( "n1-standard-1" "n1-standard-2" "n1-standard-4" )
	for i in "${array[@]}"
	do
		./create_vm_single.sh $project_name $i-vm $i $zone $service_account $startup_script
	done
fi

