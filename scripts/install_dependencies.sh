#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


# You can bypass this entire section and how long this takes by maintaining already built libraries for
# your particular distro. This section is to ensure it's functional regardless since there's no
# repo to pull already built libraries for these.
YUM_PATH="$( which yum )";
APT_PATH="$( which apt-get )";
if [ -n "$YUM_PATH" ]
then
  sudo yum update -y
  sudo amazon-linux-extras install epel -y
  sudo yum groupinstall "Development Tools"
  sudo yum install clang pulseaudio vulkan Xorg postgresql-devel -y
elif  [ -n "$APT_PATH" ]
then
  sudo apt-get update -y
  sudo apt-get install clang pulseaudio xorg postgresql vulkan-utils -y
else
   echo "This script does not support arena dependencies installation on your platform. Please install them manually."
fi

sudo cp -r "$HOME"/AlexaArena/arena/Dependencies/* /usr/lib/
sudo ldconfig

# Install python3 dependencies
pip3 install flask numpy
"$HOME"/anaconda3/envs/pytorch_p3*/bin/pip install torchvision==0.13.1

# Update GPU drivers
curl -fSsl -O https://us.download.nvidia.com/tesla/470.82.01/NVIDIA-Linux-x86_64-470.82.01.run
sudo sh NVIDIA-Linux-x86_64-470.82.01.run --ui=none --no-questions
sudo nvidia-xconfig

# Install NodeJS using NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
source ~/.nvm/nvm.sh
nvm install 14.17.5
source ~/.nvm/nvm.sh
npm install -g n
sudo -E env "PATH=$PATH" n 14.17.5

# Setting up ffmpeg
sudo rm -rf /usr/local/bin/ffmpeg && sudo rm -f /usr/bin/ffmpeg && sudo mkdir -p /usr/local/bin/ffmpeg
curl -fSsl -O https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
sudo tar -xf ffmpeg-release-amd64-static.tar.xz -C /usr/local/bin/ffmpeg
sudo cp -a /usr/local/bin/ffmpeg/ffmpeg-*/. /usr/local/bin/ffmpeg
sudo ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg

# Cleanup now unused files
sudo rm NVIDIA-Linux-x86_64-470.82.01.run
rm ffmpeg-release-amd64-static.tar.xz
