### Tracking info

Docker setup in ec2 commands to be executed

sudo apt-get update -y
sudo apt-get update

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

sudo usermod -aG docker ubuntu
newgrp docker
