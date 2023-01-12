# How to debug in Arena?

### 1. Prerequisite
1. Make sure that you have arena executable under folder ```AlexaArena/arena```.
2. Ensure that the CDF is valid.
3. Run commands ```chmod +x ./debugger/run_arena_debugger.sh``` and ```chmod +x ./debugger/run_model_debugger.sh``` to 
give execute permissions to these scripts.

---
### 2. Steps
1. Open two terminals, say T1 & T2, and connect to EC2 instance. Go to AlexaArena directory.
2. On terminal T1,
    1. Run ```cd ./debugger``` and ```./run_arena_debugger.sh```
    2. Enter the CDF file path (For example: ```/home/ec2-user/AlexaArena/data/CDFs/T2_CDFs/mission_01.json```)
    3. The script spawns the arena process and launches game in CDF
    4. Once the game is launched, a color image is saved in /tmp/ directory. The color image path is displayed on terminal.
    5. (Optional) You could run streaming server for visuals (Refer [README](../web_tool/README.md) for running streaming server)
3. On terminal T2,
    1. Activate virtual environment required for running the model ```conda activate pytorch_p38```.
    2. Go to AlexaArena directory. Run ```cd ./debugger``` and ```./run_model_debugger.sh```.
    3. It loads the model and generate actions for input utterance.
    4. Enter the directory path where you'd like to store the actions (For example: /tmp/mission_01)
4. On terminal T1,
    1. It asks to enter the **actions file path**. This actions file can be generated manually or using model_debugger.
          ```
          # Sample actions file for user utterance move backawards
       
          [
            {
               "id": "43d0a1c8-444e-11ed-8484-0ed975f73903", 
               "type": "Move", 
               "move": {"direction": "Backward", "magnitude": 1}
            }
          ]  
          ```
5. On terminal T2, generate the actions file using model debugger
    1. Enter the user utterance
    2. Enter the color image path you received from step 2.4
    3. The model should predict the actions from input utterance and color image
    4. The generated actions are stored in directory inputted in step 3.4 (Say, /tmp/mission_01/actions_1.json)
    5. Go to terminal T1
6. On terminal T1,
    1. Enter the actions file path generated in step 5.4 (For example: /tmp/mission_01/actions_1.json)
    2. View the output on browser
    3. After executing the action, error code is displayed
    4. Also, the updated color image is saved in /tmp/ directory. The new image path is displayed on terminal.
7. Please continue steps 4, 5, & 6 for further debugging
