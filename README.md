# AlexaArena

---
## 1. Introduction
Arena is a new Embodied AI platform, for robotic task completion in simulated environments. Arena has been developed 
with an objective of advancing research in Human Robot Interaction (HRI) for robot task completion challenges. 
Building embodied agents for Arena involves working on key science aspects such as Multimodal Understanding and 
Reasoning, Embodied Conversational AI, Imitation and Reinforcement Learning, Teachable AI and Robotic Task planning.

This repository includes codebase to interact with the Arena executable. It also provides several scripts to fetch dataset, 
placeholder model, and other auxiliary tools. If you are interested to learn more about or try out Arena, please 
contact [arena-admins@amazon.com](). The Arena executable is subject to a separate license that allows use for 
non-commercial purposes only.

---

## 2. Installation

### 2.1 Instance configuration for running Arena
* **Number of vCPUs**: 8
* **Number of GPUs**: 1
* **Memory**: 32 GiB
* **Storage**: 200 GiB
* **Operating system**: Amazon Linux 2

### 2.2 Steps

> 1. Login to the EC2 instance from the AWS console
> 2. Pull AlexaArena repository from GitHub: [https://github.com/amazon-science/alexa-arena]()
> 3. Copy Arena zip(received via email) to your machine. Unzip the folder and extract binaries in the folder named 
"arena" (path: "AlexaArena/arena/")
> 4. Run "./scripts/install_dependencies.sh"
> 5. Once the script is finished, go to "AlexaArena/arena_installation_test" folder
> 6. Activate the pytorch_p38 conda environment. Run "conda activate pytorch_p38"
> 7. Run "./run_linux.sh". You should see "Arena dependencies installation test is completed successfully" if the installation is successful.

Note: The installation scripts mentioned above are tested on AWS EC2 instances (instance type: g4dn.2xlarge). If you 
plan to use different cloud based instance or local machine, the installation steps may vary.

---

## 3. Data & Baseline Models
We provide two separate datasets to assist model training. The first dataset contains trajectory data with robot action trajectories annotated with human natural language instructions and question-answers. It can be used for training and evaluating robot models for task completion. The second dataset contains image data generated via Arena that can be used to train amd evaluate vision models that can work in Arena environment. Please find the detailed information about the data and how to download them [here](data/trajectory-data/README.md) and [here](data/vision-data/README.md).

We also provide several baseline models for robot task completion. Please find detailed information [here](modeling/README.md)

---
## 4. Auxiliary Tools

In addition to Arena simulator and model training/evaluation scripts, this package includes some auxiliary tools that 
could help during model development. Please find them below.

### 4.1. Chat-based Interactive WebTool
It is a web application that enables users to communicate with the robot in a simulated environment. It enables the 
control of an AI agent to alter object's state in real-time using a chat interface. It provides an ability to launch 
a game in CDF (Challenge Definition Format) file and complete it using natual language commands. Each command is sent to
a model for generating a sequence of primitive actions. Following that, the Arena executable processes these actions and 
outcome is shown on web browser. Please find the web-tool UI below.

![WebTool UI](images/place-control-panel.png)

To learn more about it, please click [here](./web_tool/README.md).

### 4.2. Arena Debugger
Arena debugger is a software tool that could be used to test and debug end-to-end workflow. There are three critical steps 
in debugging a model for Arena
1. Process the input utterance and predict actions & object classes
2. Predict mask for objects
3. Generate a JSON using predicted mask and actions with required format

This tool allows developer to pause and inspect the outcome after every step. Please check the 
[Arena Debugger README](./debugger/README.md) for more information.

---

## 5. Eval AI Challenge
The Alexa Arena Challenge is available on Eval AI platform. Please find more details here: https://eval.ai/web/challenges/challenge-page/1903/overview

For the challenge, this module offers the required code snippet to produce metadata output files. More information is available [here](./eval_ai/README.md)

---

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the LGPL-2.1 License.
