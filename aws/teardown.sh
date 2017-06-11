#!/bin/bash

source check.sh $1

# Check if env exists
if [ ! -f $settings ]; then
   echo "Environment $settings does not exist."
   exit 0
fi

source $settings

aws ec2 release-address --allocation-id $allocId
aws ec2 delete-security-group --group-id $securityGroupId
aws ec2 disassociate-route-table --association-id $routeTableAssoc
aws ec2 delete-route-table --route-table-id $routeTableId
aws ec2 detach-internet-gateway --internet-gateway-id $gatewayId --vpc-id $vpcId
aws ec2 delete-internet-gateway --internet-gateway-id $gatewayId
aws ec2 delete-subnet --subnet-id $subnetId
aws ec2 delete-vpc --vpc-id $vpcId
echo If you want to delete $keyId, please do it manually.
