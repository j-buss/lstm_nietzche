#!/bin/bash
echo Script name: $0
echo $# arguments 
if [ $# -ne 1 ]; then
        echo "illegal number of parameters"
        echo "Arguments: project_name"
else
        project_name=$1
	TOKEN=$(gcloud auth application-default print-access-token)
	AUTHORIZATION_STRING='Authorization: Bearer '
	curl --request POST \
  	'https://monitoring.googleapis.com/v3/projects/'$project_name'/alertPolicies' \
  	--header "$AUTHORIZATION_STRING$TOKEN" \
  	--header 'Accept: application/json' \
  	--header 'Content-Type: application/json' \
  	--data '{"displayName":"Low CPU Warning","combiner":"OR","conditions":[{"displayName":"CPU Utilization is Low","conditionThreshold":{"aggregations":[{"alignmentPeriod":"60s","crossSeriesReducer":"REDUCE_MEAN","groupByFields":["project","resource.label.instance_id","resource.label.zone"],"perSeriesAligner":"ALIGN_MAX"}],"comparison":"COMPARISON_LT","duration":"600s","filter":"metric.type=\"compute.googleapis.com/instance/cpu/utilization\" AND resource.type=\"gce_instance\"","thresholdValue":0.09,"trigger":{"count":1}}}]}' \
  	--compressed
fi	
