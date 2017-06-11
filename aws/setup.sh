#!/bin/bash

source check.sh $1

# Check if env exists
if [ -f $settings ]; then
   echo "Environment $settings exists."
   exit 0
fi

source config.sh

# settings
export cidr="0.0.0.0/0"

echo Setting up vpc...
export vpcId=`aws ec2 create-vpc --cidr-block 10.0.0.0/28 --query 'Vpc.VpcId' --output text`
aws ec2 create-tags --resources $vpcId --tags --tags Key=Name,Value=$name
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-support "{\"Value\":true}"
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-hostnames "{\"Value\":true}"
echo vpcId=$vpcId

echo Setting up gateway...
export gatewayId=`aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text`
aws ec2 create-tags --resources $gatewayId --tags --tags Key=Name,Value=$name-gateway
aws ec2 attach-internet-gateway --internet-gateway-id $gatewayId --vpc-id $vpcId
echo gatewayId=$gatewayId

echo Setting up subnet...
export subnetId=`aws ec2 create-subnet --vpc-id $vpcId --cidr-block 10.0.0.0/28 --availability-zone $zone --query 'Subnet.SubnetId' --output text`
aws ec2 modify-subnet-attribute --subnet-id $subnetId --map-public-ip-on-launch
aws ec2 create-tags --resources $subnetId --tags --tags Key=Name,Value=$name-subnet
echo subnetId=$subnetId

echo Setting up route table...
export routeTableId=`aws ec2 create-route-table --vpc-id $vpcId --query 'RouteTable.RouteTableId' --output text`
aws ec2 create-tags --resources $routeTableId --tags --tags Key=Name,Value=$name-route-table
export routeTableAssoc=`aws ec2 associate-route-table --route-table-id $routeTableId --subnet-id $subnetId --output text`
aws ec2 create-route --route-table-id $routeTableId --destination-cidr-block 0.0.0.0/0 --gateway-id $gatewayId
echo routeTableId=$routeTableId

echo Setting up security group...
export securityGroupId=`aws ec2 create-security-group --group-name $name-security-group --description "$name-security-group" --vpc-id $vpcId --query 'GroupId' --output text`
# ssh
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 22 --cidr $cidr
# jupyter notebook
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 8888-8898 --cidr $cidr
echo securityGroupId=$securityGroupId

echo Setting up elastic ip
export allocId=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
echo allocId=$allocId

if [ ! -d ~/.ssh ]
then
    mkdir ~/.ssh
fi

if [ ! -f ~/.ssh/aws-key-$name.pem ]
then
    aws ec2 create-key-pair --key-name aws-key-$name --query 'KeyMaterial' --output text > ~/.ssh/aws-key-$name.pem
    chmod 400 ~/.ssh/aws-key-$name.pem
fi

cat >$settings <<EOF
export name=$name
export region=$region
export vpcId=$vpcId
export subnetId=$subnetId
export gatewayId=$gatewayId
export routeTableId=$routeTableId
export routeTableAssoc=$routeTableAssoc
export securityGroupId=$securityGroupId
export allocId=$allocId
export keyId=aws-key-$name
EOF
