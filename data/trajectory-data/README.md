## Dataset overview

This dataset contains the ground-truth action trajectories for completing 3k+ game missions. The action trajectories are also paired with robot view images. Each trajectory is annotated with 3 sets of language instructions. And on top of each set of human language instruction, there are 2 sets of questions and answers collected. Other than human language annotations, for each action trajectory we also provide a set of template-based synthetic language instructions. 

## Download

Run `./scripts/fetch_trajectory_data.sh` to download the data from S3 bucket.

## Data file structure

```
{
  "<game_mission_id>": {                                (unique game mission id)
    "human_annotations": [                              (human instruction and QA annotations)
      {
        "instructions": [
          {
            "instruction": "go to the round table",
            "actions": [0,1],                           (action index corresponding to the instruction)
            "question_answers": [
              {
                "question": "Where is the table?",
                "answer": "the table is behind you",
                "question_necessary": true              (whether the annotator thinks asking this question is necessary)
              },
              ...
            ]
          },
          ...
        ]
      },
      ...
    ],
    "synthetic_annotations": [                          (template-based instructions)
      {
        "instructions": [
          {
            "instruction": "go to the robotics lab",
            "actions": [0]                              (action index corresponding to the instruction)
          },
          ...
        ]
      },
      ...
    ],
    "actions": [                                        (robot actions)
      {
        "id": 0,                                        (action index)
        "type": "Look",                                 (action type)
        "colorImages": [                                
          "<image_filename>"                            (robot view image file)
        ],
        "look": {
          "direction": "Around",                        (look_around action)
          "magnitude": 100                              (field of view in degrees)
        }
      },
      {
        "id": 1,
        "type": "Goto",
        "colorImages": [                                (multiple robot view images)
          "<image_filename>",
          "<image_filename>",
          "<image_filename>",
          "<image_filename>"
        ],
        "goto": {
          "object": {
            "id": "Table_01",                           (object id)
            "colorImageIndex": 3,                       (indicating which color image has the object mask)
            "mask": [[50037, 10], ..., [53938, 7]]      (compressed object mask)
          }
        }
      },
      ...
    ],
    "CDF": {...}                                        (CDF is used in Arena to load the game scene)
  },
  ...
}
```

Image files can be find in `/mission_images` with a subfolder named with the corresponding `game_mission_id`. 