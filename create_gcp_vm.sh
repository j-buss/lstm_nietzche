#!/bin/bash
echo Script name: $0
echo $# arguments 
if [ $# -ne 3 ]; then
	echo "illegal number of parameters"
	echo "Arguments: project_name instance_name machine_type"
else
	project_name=$1
	instance_name=$2
	machine_type=$3

	gcloud beta compute --project=$project_name instances create $instance_name \
	--zone=us-central1-c \
	--machine-type=$machine_type \
	--subnet=default \
	--network-tier=PREMIUM \
	--metadata=startup-script-url=gs://config-001/test.sh \
	--maintenance-policy=MIGRATE \
	--service-account=548848141143-compute@developer.gserviceaccount.com \
	--scopes=https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/devstorage.read_write \
       	--image-family=debian-9 \
	--image-project=debian-cloud \
	--boot-disk-size=10GB \
	--boot-disk-type=pd-standard \
	--boot-disk-device-name=$instance_name
fi
