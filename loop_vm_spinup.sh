#!/bin/bash
echo Script name: $0
echo $# arguments
project_name=$1
if [ $# -ne 1 ]; then
        echo "illegal number of parameters"
        echo "Argument: project_name" 
else
	array=( "f1-micro" "n1-standard-1" "n1-standard-2" "n1-standard-4" "n1-standard-8" )
	for i in "${array[@]}"
	do
		./create_gcp_vm.sh $project_name $i-vm $i
	done
fi
