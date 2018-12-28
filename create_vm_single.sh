#!/bin/bash
echo Script name: $0
echo $# arguments 
if [ $# -ne 6 ]; then
	echo "illegal number of parameters"
	echo "Arguments: project_name instance_name machine_type zone service_account startup_script"

else
	project_name=$1
	instance_name=$2
	machine_type=$3
	zone=$4
	service_account=$5
	startup_script=$6

	gcloud beta compute --project=$project_name instances create $instance_name \
	--zone=$zone \
	--machine-type=$machine_type \
	--subnet=default \
	--network-tier=PREMIUM \
	--metadata=startup-script-url=$startup_script \
	--maintenance-policy=MIGRATE \
	--service-account=$service_account \
	--scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
       	--image-family=debian-9 \
	--image-project=debian-cloud \
	--boot-disk-size=10GB \
	--boot-disk-type=pd-standard \
	--boot-disk-device-name=$instance_name
fi
