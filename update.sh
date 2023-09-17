#!/bin/bash

apt-get -y update

apt-get install -y  --reinstall \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
    
mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
  
apt-get -y update

apt-get install -y --reinstall docker-ce docker-ce-cli containerd.io docker-compose-plugin