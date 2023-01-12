# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


########################################################################################################################
# The list of object classes that can be used with OBJECT_CLASS as ObjectOutputType.

OBJECT_CLASS_ALLOW_LIST = ["stickynote"]
########################################################################################################################
# Action space

NAVIGATIONAL_ACTIONS = ["Goto", "Move", "Rotate", "Look"]
OBJECT_INTERACTION_ACTIONS = ["Pickup", "Open", "Close", "Break", "Scan", "Examine", "Place", "Pour", "Toggle",
                              "Fill", "Clean"]
ACTIONS_REQUIRING_MASK = ["Pickup", "Open", "Close", "Break", "Scan", "Examine", "Place", "Pour", "Toggle", "Fill",
                          "Clean", "Goto"]
########################################################################################################################
