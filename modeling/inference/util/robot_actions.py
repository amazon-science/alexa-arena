# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import uuid


OBJECT_INTERACT_ACTIONS = {'Toggle', 'Pickup', 'Place', 'Open', 'Close', 'Pour', 'Fill', 'Clean', 'Break', 'Examine', 'Scan'}

# Robot action commands
INTERACT_ACTION = {
    "id": str(uuid.uuid1()),
    "type": "", 
    "dummy_act": {
        "object": {
            "colorImageIndex": 0,
            "name": "",
            "mask": None
        }
    }
}

GOTO_ROOM_ACTION = {
    "id": "",
    "type": "Goto",
    "goto": {
        "object": {
            "officeRoom": ""
        }
    }
}

GOTO_VIEWPOINT_ACTION = {
    "id": "",
    "type": "Goto",
    "goto": {
        "object": {
            "goToPoint": "",
        }
    }
}

GOTO_OBJECT_ACTION = {
    "id": "",
    "type": "Goto",
    "goto": {
        "object": {
            "colorImageIndex": 0,
            "name": "",
        }
    }
}

LOOK_AROUND_ACTION = {
    "id": "",
    "type": "Look", 
    "look": {
        "direction": "Around",
        "magnitude": 100
    }
}

LOOK_ACTION = {
    "id": "",
    "type": "Look",
    "look": {
        "direction": "",  # Up, Down
        "magnitude": 45.0,
    }
}

MOVE_ACTION = {
    "id": "", 
    "type": "Move", 
    "move": {
        "direction": "",  # Forward or Backward
        "magnitude": 1,
    }
}

ROTATE_ACTION = {
    "id": "",
    "type": "Rotate", 
    "rotation": {
        "direction": "",  # Right or Left
        "magnitude": 45.0,
    }
}
