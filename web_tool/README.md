# Chat-based Interactive WebTool
It is a web application that enables users to communicate with the robot in a simulated environment. It enables the
control of an AI agent to alter object's state in real-time using a chat interface. It provides an ability to launch
a game in CDF (Challenge Definition Format) file and complete it using natual language commands. Each command is sent to
a model for generating a sequence of primitive actions. Following that, the Arena executable processes these actions and
outcome is shown on web browser.

---

## How to run web-tool on your instance?

### 1. Run StreamingServer

#### 1.1 Pre-requisites

Ensure that your EC2 instance has 81, 3000, 11000, and 19302 as an inbound port in the security group.
1. Go to the EC2 Instance Console.
2. Go to the security tab for your instance.
3. Click the Security Group to edit it.
4. Click `Edit inbound rules` to add a rule to allow TCP Traffic over port 81, 3000, 11000, and 19302 from 0.0.0.0/0
5. Hit `Save rules`

#### 1.2 Running the streaming server

1. Open terminal T1, go to AlexaArena directory and run ```./scripts/run_streaming_server.sh```.
2. Use the Public IP Address and then go to a browser to connect to the streaming server using the following 
address: `http://<PUBLIC_IP>:81` and hit play.

### 2. Run frontend and backend servers
1. Open terminal T2, run frontend server using following commands:
    1. Go to front end directory ```cd AlexaArena/web_tool/frontend```
    2. Run ```npm install```
    3. Run ```export PUBLIC_DNS=$(curl -s http://169.254.169.254/latest/meta-data/public-hostname)```
    4. ADD public IP of ec2 instance in ```AlexaArena/web_tool/front-end/src/components/Urls.jsx```
    5. Run ```npm start -- --host=$PUBLIC_DNS```
2. Open terminal T3,
    1. Run "AlexaArena/scripts/fetch_vision_model.sh"
    2. Run "conda activate pytorch_p38"
    3. Run backend server using ```cd $HOME/AlexaArena/web_tool && ./run_server.sh```

---
