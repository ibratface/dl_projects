#!/bin/bash

source check.sh
source config.sh
source $settings

cat >$name-$region-spec.json <<EOF
{
  "ImageId" : "$amiId",
  "InstanceType": "$instanceType",
  "Placement": {
      "AvailabilityZone": "$zone"
  },
  "KeyName" : "$keyId",
  "EbsOptimized": true,
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "DeleteOnTermination": true,
        "VolumeType": "gp2",
        "VolumeSize": 128
      }
    }
  ],
  "NetworkInterfaces": [
      {
        "DeviceIndex": 0,
        "SubnetId": "${subnetId}",
        "Groups": [ "${securityGroupId}" ],
        "AssociatePublicIpAddress": false
      }
  ]
}
EOF

# common commands
alias aws-attach='aws ec2 attach-volume --volume-id $volumeId --instance-id $instanceId --device $deviceId && aws ec2 wait volume-in-use --volume-ids $volumeId'
alias aws-detach='aws ec2 detach-volume --volume-id $volumeId --instance-id $instanceId --device $deviceId'
alias aws-ip='export instanceIp=`aws ec2 describe-instances --filters "Name=instance-id,Values=$instanceId" --query "Reservations[0].Instances[0].PublicIpAddress"` && echo $instanceIp'
alias aws-request-spot='export spotRequestId=`aws ec2 request-spot-instances --launch-specification file://$name-$region-spec.json --spot-price $bidPrice --output="text" --query="SpotInstanceRequests[*].SpotInstanceRequestId"`'
alias aws-wait-spot='aws ec2 wait spot-instance-request-fulfilled --spot-instance-request-ids $spotRequestId'
alias aws-get-spot='export instanceId=`aws ec2 describe-spot-instance-requests --spot-instance-request-ids $spotRequestId --query="SpotInstanceRequests[*].InstanceId" --output="text"`'
alias aws-spot='aws-request-spot && aws-wait-spot && aws-get-spot'
alias aws-spot-addr='aws ec2 associate-address --instance-id $instanceId --allocation-id $allocId'
alias aws-start='aws-spot-addr && aws ec2 wait instance-running --instance-ids $instanceId && aws-attach && aws-ip'
alias aws-stop='aws ec2 terminate-instances --instance-ids $instanceId && aws ec2 wait instance-terminated --instance-ids $instanceId && ssh-keygen -R $instanceIp'

# login
alias aws-ssh='ssh -i ~/.ssh/aws-key-fast-ai.pem $user@$instanceIp'
