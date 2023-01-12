## CDF

We use CDF (Challenge Definition Format) files to store game mission information. One CDF file is a json-format file that is readable by Arena. It contains all necessary information to run a game mission in Arena, including:
* Game initial state: game scene, robot initial location and state, object initial location and initial states (such as cabinet door open or close)
* Game goal state: robot and object locations and states that is necessary to complete the game mission (Once all the goal states are met, the game mission is considered completed)
* Game-related text data: game mission description text, etc.