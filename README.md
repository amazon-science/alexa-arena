# AlexaArena

---
## 1. Introduction
Arena is a new Embodied AI platform, for robotic task completion in simulated environments. Arena has been developed 
with an objective of advancing research in Human Robot Interaction (HRI) for robot task completion challenges. 
Building embodied agents for Arena involves working on key science aspects such as Multimodal Understanding and 
Reasoning, Embodied Conversational AI, Imitation and Reinforcement Learning, Teachable AI and Robotic Task planning.

This repository includes codebase to interact with the Arena executable. It also provides several scripts to fetch 
dataset, placeholder model, and other auxiliary tools. The Arena executable is not available in this repository. If 
you are interested to learn more about Arena or request access to the Arena executable, please contact 
[arena-admins@amazon.com](). The Arena executable is subject to a separate 
[license](ARENA_EXECUTABLE_LICENSE) that allows use for non-commercial purposes only.

---

## 2. Installation

### 2.1 Instance configuration for running Arena
* **Number of vCPUs**: 8
* **Number of GPUs**: 1
* **Memory**: 32 GiB
* **Storage**: 200 GiB
* **Operating system**: Amazon Linux 2(ami-0496aeed90a040b1b), Ubuntu 18.04(ami-0475f1fb0e9b1f73f)

Please refer to [this](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html) tutorial for information
on how to create an AWS EC2 instance with the aforementioned configuration.

### 2.2 Steps

> 1. Login to the EC2 instance from the AWS console
> 2. Pull AlexaArena repository from GitHub to $HOME directory: [https://github.com/amazon-science/alexa-arena]() ```git clone https://github.com/amazon-science/alexa-arena.git AlexaArena``` (note to clone into AlexaArena for script compatibility)
> 3. If you have not done so already, download aws cli from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
> 4. Copy ```fetch_arena.sh```(received via email) to path ```"$HOME/AlexaArena/scripts/"``` and run it. This script 
would download and extract arena binaries in folder: ```"$HOME/AlexaArena/arena/"```
> 5. Run "$HOME/AlexaArena/scripts/install_dependencies.sh"
> 6. Once the script is finished, go to "AlexaArena/arena_installation_test" folder
> 7. Run "./run_linux.sh". You should see "Arena dependencies installation test is completed successfully" if the 
installation is successful.

**Note**: The installation script mentioned above is tested on AWS EC2 instances [Instance types: g4dn.2xlarge, 
p3.8xlarge OS: Amazon Linux 2, Ubuntu 18.04]. If you plan to use different cloud based instance 
or local machine, the installation steps may vary. Please refer [this](scripts/install_dependencies.sh) to know 
about dependencies.

---

## 3. Data & Baseline Models
We provide two separate datasets to assist model training, which are made available under the CC BY-NC 4.0 [license](DATA_LICENSE). The 
first dataset contains trajectory data with robot action trajectories annotated with human natural language instructions 
and question-answers. It may be useful for training and evaluating robot models for task completion. The second dataset 
contains image data generated via Arena, and it may be useful for training and evaluating vision models that can work in 
the Arena environment. Please find the detailed information about the data and how to download them 
[here](data/trajectory-data/README.md) and [here](data/vision-data/README.md).

We also provide baseline models for robot task completion. Please find detailed information [here](modeling/README.md)

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

For the challenge, this module offers the required code snippet to produce metadata output files. More information is 
available [here](./eval_ai/README.md)

---

## 6. Citation
Alexa Arena has been used in:

- Alexa Arena: A User-Centric Interactive Platform for Embodied AI. [PDF](https://arxiv.org/pdf/2303.01586) <br/>
Gao, Q., Thattai, G., Gao, X., Shakiah, S., Pansare, S., Sharma, V., ... & Natarajan, P. <br/>
arXiv preprint arXiv:2303.01586.

If you use the platform, please consider citing our paper. 

```
@article{gao2023alexa,
  title={Alexa Arena: A User-Centric Interactive Platform for Embodied AI},
  author={Gao, Qiaozi and Thattai, Govind and Gao, Xiaofeng and Shakiah, Suhaila and Pansare, Shreyas and Sharma, Vasu and Sukhatme, Gaurav and Shi, Hangjie and Yang, Bofei and Zheng, Desheng and others},
  journal={arXiv preprint arXiv:2303.01586},
  year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the LGPL-2.1 License.
