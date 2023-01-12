# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import numpy as np
import logging

from arena_wrapper.enums.object_output_wrapper import ObjectOutputType
from arena_wrapper.util import decompress_mask


class RGActionsConstant:
    DEFAULT_ROTATE_MAGNITUDE = 45
    DEFAULT_MOVE_MAGNITUDE = 1
    DEFAULT_LOOK_MAGNITUDE = 45
    DEFAULT_LOOK_AROUND_MAGNITUDE = 100


class ArenaRequestBuilder:
    def __init__(self):
        self.logger = logging.getLogger("ArenaRequestBuilder")
        self.ground_truth_segmentation_images = None
        self.segmentation_color_to_object_id_map = None
        self.objects_in_hands = {"left": None, "right": None}

    def find_object_id(self, compressed_mask, color_image_index):
        ## Decompress the mask
        mask = decompress_mask(compressed_mask)
        if self.ground_truth_segmentation_images is None or not self.ground_truth_segmentation_images:
            self.logger.error("Unable to find the object id, previous segmentation images are not present")
        ## Use the first segmentation image until InferenceService specifies which image to use along with the mask
        ground_truth_segmentation_image = self.ground_truth_segmentation_images[color_image_index]

        ## Perform element wise multiplication of mask_matrix and previous_seg_image
        mask_3d = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
        masked_image = ground_truth_segmentation_image * mask_3d

        ## Make a list of all colors present in result_image
        unique_colors = np.unique(masked_image.reshape(-1, masked_image.shape[2]), axis=0)

        ## Compute IoU for each color and find the color that has maximum IoU.
        pred_indices = np.where(np.all(mask_3d == 1, axis=-1))
        pred_indices = set([(x, y) for x, y in zip(*pred_indices)])
        ious = []
        for i in range(unique_colors.shape[0]):
            color = tuple(unique_colors[i])
            indices = np.where(np.all(ground_truth_segmentation_image == color, axis=-1))
            indices = set([(x, y) for x, y in zip(*indices)])
            intersection = pred_indices.intersection(indices)
            union = pred_indices.union(indices)
            ious.append(len(intersection) / len(union))
        ious = np.array(ious)
        max_ind = np.argmax(ious)
        max_color = unique_colors[max_ind]
        self.logger.info("Segmentation color with maximum IoU: " + str(max_color))

        ## Determine object id based on identified color using instance Segmentation color to object id map
        key = frozenset({'r': int(max_color[0]), 'g': int(max_color[1]), 'b': int(max_color[2])}.items())
        if key in self.segmentation_color_to_object_id_map:
            object_id = self.segmentation_color_to_object_id_map[key]
            self.logger.info("Found object id: " + str(object_id))
        else:
            self.logger.error("Unable to find the object id")
            object_id = None

        return object_id

    def get_request_json(self, action_request, params):
        rg_compatible_request = None
        self.ground_truth_segmentation_images = params['segmentationImages']
        self.segmentation_color_to_object_id_map = params['segmentationColorToObjectIdMap']
        self.object_output_type = params['objectOutputType']
        self.objects_in_hands["right"] = params["rightHandHeldObject"]
        if action_request["type"] == "Move":
            rg_compatible_request = self._build_move_request(action_request["move"])
        elif action_request["type"] == "Rotate":
            rg_compatible_request = self._build_rotate_request(action_request["rotation"])
        elif action_request["type"] == "Goto":
            rg_compatible_request = self._build_goto_request(action_request["goto"])
        elif action_request["type"] == "Pickup":
            rg_compatible_request = self._build_pickup_request(action_request["pickup"])
        elif action_request["type"] == "Place":
            rg_compatible_request = self._build_place_request(action_request["place"])
        elif action_request["type"] == "Open":
            rg_compatible_request = self._build_open_request(action_request["open"])
        elif action_request["type"] == "Close":
            rg_compatible_request = self._build_close_request(action_request["close"])
        elif action_request["type"] == "Break":
            rg_compatible_request = self._build_break_request(action_request["break"])
        elif action_request["type"] == "Scan":
            rg_compatible_request = self._build_scan_request(action_request["scan"])
        elif action_request["type"] == "Pour":
            rg_compatible_request = self._build_pour_request(action_request["pour"])
        elif action_request["type"] == "Toggle":
            rg_compatible_request = self._build_toggle_request(action_request["toggle"])
        elif action_request["type"] == "Throw":
            rg_compatible_request = self._build_throw_request(action_request["throw"])
        elif action_request["type"] == "Fill":
            rg_compatible_request = self._build_fill_request(action_request["fill"])
        elif action_request["type"] == "Clean":
            rg_compatible_request = self._build_clean_request(action_request["clean"])
        elif action_request["type"] == "Examine":
            rg_compatible_request = self._build_examine_request(action_request["examine"])
        elif action_request["type"] == "CameraChange":
            rg_compatible_request = self._build_camera_change_request(action_request["camerachange"])
        elif action_request["type"] == "Look":
            rg_compatible_request = self._build_look_request(action_request["look"])
        elif action_request["type"] == "Highlight":
            rg_compatible_request = self._build_highlight_request(action_request["highlight"])
        else:
            self.logger.error("Incorrect action format received." + str(action_request))
            raise ValueError("Invalid action dictionary received")
        return rg_compatible_request

    def _build_move_request(self, input_move_request):
        move_request = {}
        if input_move_request["direction"] == "Forward":
            move_request["commandType"] = "MoveForward"
        elif input_move_request["direction"] == "Backward":
            move_request["commandType"] = "MoveBackward"
        if "magnitude" not in input_move_request:
            move_request["magnitude"] = RGActionsConstant.DEFAULT_MOVE_MAGNITUDE
        else:
            move_request["magnitude"] = input_move_request["magnitude"]
        return move_request

    def _build_rotate_request(self, input_rotate_request):
        rotate_request = {"commandType": "Rotate"}
        rotation_angle = RGActionsConstant.DEFAULT_ROTATE_MAGNITUDE
        if "magnitude" in input_rotate_request:
            rotation_angle = input_rotate_request["magnitude"]
        if input_rotate_request["direction"] == "Left":
            rotate_request["magnitude"] = -rotation_angle
        elif input_rotate_request["direction"] == "Right":
            rotate_request["magnitude"] = rotation_angle
        if "rotationSpeed" in input_rotate_request:
            rotate_request["rotationSpeed"] = input_rotate_request["rotationSpeed"]
        return rotate_request

    def _build_goto_request(self, input_goto_request):
        # "position" and "raycast" will be set later
        goto_request = self._get_goto_command_json()
        if "object" not in input_goto_request:
            self.logger.error(f"Input request did not contain \"object\": {input_goto_request}")
            goto_request = None
        elif "officeRoom" in input_goto_request["object"]:
            self.logger.info(f"Using \"officeRoom\" in goTo command")
            goto_request["goToCommand"]["officeRoom"] = input_goto_request["object"]["officeRoom"]
        elif "goToPoint" in input_goto_request["object"]:
            self.logger.info(f"Using \"goToPoint\" in goTo command")
            goto_request["goToCommand"]["goToPoint"] = input_goto_request["object"]["goToPoint"]
        elif "mask" in input_goto_request["object"] and self.object_output_type == ObjectOutputType.OBJECT_MASK:
            self.logger.info(f"Using \"mask\" in goTo command")
            color_image_index = input_goto_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_goto_request["object"]["mask"], color_image_index)
            goto_request["goToCommand"]["instanceIdOfObject"] = object_id
        elif "name" in input_goto_request["object"]:
            self.logger.info(f"Using \"name\" in goTo command")
            goto_request["goToCommand"]["instanceIdOfObject"] = input_goto_request["object"]["name"]
        else:
            self.logger.error(f"Did not find required goTo parameters in \"object\": {input_goto_request}")
            goto_request = None
        return goto_request

    def _get_interact_command_json(self):
        return {"commandType": "Interact", "interactCommand": {"sourceObjectID": "TAM_1", "destinationObjectID": "", "verb": ""}}

    def _get_pickup_command_json(self):
        return {"commandType": "PickUp", "pickUpOrPlaceCommand": {"destinationObjectID": "", "useLeftHand": "false"}}

    def _get_place_command_json(self):
        return {"commandType": "Place", "pickUpOrPlaceCommand": {"destinationObjectID": "", "useLeftHand": "false"}}

    def _get_camera_change_command_json(self):
        return {"commandType": "CameraChange", "camChangeCommand": {"mode": ""}}

    def _get_goto_command_json(self):
        return {"commandType": "GoTo", "goToCommand": {"instanceIdOfObject": None, "position": None, "raycast": None,
                                                       "officeRoom": None, "goToPoint": None}}
    def _get_highlight_command_json(self):
        return {"commandType": "Highlight", "highlightCommand": {"instanceId": None, "shouldRotateToObject": True,
                                                                 "shouldRotateBack": True}}

    def _build_pickup_request(self, input_pickup_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_pickup_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_pickup_request["object"]["mask"], color_image_index)
            input_pickup_request["object"]["name"] = object_id
        pickup_request = self._get_pickup_command_json()
        pickup_request["pickUpOrPlaceCommand"]["destinationObjectID"] = input_pickup_request["object"]["name"]
        return pickup_request

    def _build_place_request(self, input_place_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_place_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_place_request["object"]["mask"], color_image_index)
            input_place_request["object"]["name"] = object_id
        place_request = self._get_place_command_json()
        place_request["pickUpOrPlaceCommand"]["destinationObjectID"] = input_place_request["object"]["name"]
        return place_request

    def _build_open_request(self, input_open_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_open_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_open_request["object"]["mask"], color_image_index)
            input_open_request["object"]["name"] = object_id
        open_request = self._get_interact_command_json()
        open_request["interactCommand"]["destinationObjectID"] = input_open_request["object"]["name"]
        open_request["interactCommand"]["verb"] = "OPEN"
        return open_request

    def _build_close_request(self, input_close_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_close_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_close_request["object"]["mask"], color_image_index)
            input_close_request["object"]["name"] = object_id
        close_request = self._get_interact_command_json()
        close_request["interactCommand"]["destinationObjectID"] = input_close_request["object"]["name"]
        close_request["interactCommand"]["verb"] = "CLOSE"
        return close_request

    def _build_break_request(self, input_break_request):
        if self.objects_in_hands["right"] is not None:
            source_object_id = self.objects_in_hands["right"]
        else:
            source_object_id = "TAM_1"
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_break_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_break_request["object"]["mask"], color_image_index)
            input_break_request["object"]["source"] = source_object_id
            input_break_request["object"]["destination"] = object_id
        elif self.object_output_type == ObjectOutputType.OBJECT_CLASS and "name" in input_break_request["object"]:
            input_break_request["object"]["source"] = source_object_id
            input_break_request["object"]["destination"] = input_break_request["object"]["name"]
        break_request = self._get_interact_command_json()
        break_request["interactCommand"]["sourceObjectID"] = input_break_request["object"]["source"]
        break_request["interactCommand"]["destinationObjectID"] = input_break_request["object"]["destination"]
        break_request["interactCommand"]["verb"] = "BREAK"
        return break_request

    def _build_scan_request(self, input_scan_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_scan_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_scan_request["object"]["mask"], color_image_index)
            input_scan_request["object"]["name"] = object_id
        scan_request = self._get_interact_command_json()
        scan_request["interactCommand"]["destinationObjectID"] = input_scan_request["object"]["name"]
        scan_request["interactCommand"]["verb"] = "SCAN"
        return scan_request

    def _build_pour_request(self, input_pour_request):
        if self.objects_in_hands["right"] is not None:
            source_object_id = self.objects_in_hands["right"]
        else:
            source_object_id = input_pour_request["object"]["source"] = "TAM_1"
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_pour_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_pour_request["object"]["mask"], color_image_index)
            input_pour_request["object"]["source"] = source_object_id
            input_pour_request["object"]["destination"] = object_id
        elif self.object_output_type == ObjectOutputType.OBJECT_CLASS and "name" in input_pour_request["object"]:
            input_pour_request["object"]["source"] = source_object_id
            input_pour_request["object"]["destination"] = input_pour_request["object"]["name"]
        pour_request = self._get_interact_command_json()
        pour_request["interactCommand"]["sourceObjectID"] = input_pour_request["object"]["source"]
        pour_request["interactCommand"]["destinationObjectID"] = input_pour_request["object"]["destination"]
        pour_request["interactCommand"]["verb"] = "POUR"
        return pour_request

    def _build_throw_request(self, input_throw_request):
        if self.objects_in_hands["right"] is not None:
            source_object_id = self.objects_in_hands["right"]
        else:
            source_object_id = "TAM_1"
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_throw_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_throw_request["object"]["mask"], color_image_index)
            input_throw_request["object"]["source"] = source_object_id
            input_throw_request["object"]["destination"] = object_id
        elif self.object_output_type == ObjectOutputType.OBJECT_CLASS and "name" in input_throw_request["object"]:
            input_throw_request["object"]["source"] = source_object_id
            input_throw_request["object"]["destination"] = input_throw_request["object"]["name"]
        throw_request = self._get_interact_command_json()
        throw_request["interactCommand"]["sourceObjectID"] = input_throw_request["object"]["source"]
        throw_request["interactCommand"]["destinationObjectID"] = input_throw_request["object"]["destination"]
        throw_request["interactCommand"]["verb"] = "THROW"
        return throw_request

    def _build_clean_request(self, input_clean_request):
        if self.objects_in_hands["right"] is not None:
            source_object_id = self.objects_in_hands["right"]
        else:
            source_object_id = "TAM_1"
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_clean_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_clean_request["object"]["mask"], color_image_index)
            input_clean_request["object"]["source"] = source_object_id
            input_clean_request["object"]["destination"] = object_id
        elif self.object_output_type == ObjectOutputType.OBJECT_CLASS and "name" in input_clean_request["object"]:
            input_clean_request["object"]["source"] = source_object_id
            input_clean_request["object"]["destination"] = input_clean_request["object"]["name"]
        clean_request = self._get_interact_command_json()
        clean_request["interactCommand"]["sourceObjectID"] = input_clean_request["object"]["source"]
        clean_request["interactCommand"]["destinationObjectID"] = input_clean_request["object"]["destination"]
        clean_request["interactCommand"]["verb"] = "CLEAN"
        return clean_request

    def _build_fill_request(self, input_fill_request):
        destination_object_id = None
        if self.objects_in_hands["right"] is not None:
            destination_object_id = self.objects_in_hands["right"]
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_fill_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_fill_request["object"]["mask"], color_image_index)
            input_fill_request["object"]["destination"] = destination_object_id
            input_fill_request["object"]["source"] = object_id
        elif self.object_output_type == ObjectOutputType.OBJECT_CLASS and "name" in input_fill_request["object"]:
            input_fill_request["object"]["source"] = input_fill_request["object"]["name"]
            input_fill_request["object"]["destination"] = destination_object_id
        fill_request = self._get_interact_command_json()
        fill_request["interactCommand"]["sourceObjectID"] = input_fill_request["object"]["source"]
        fill_request["interactCommand"]["destinationObjectID"] = input_fill_request["object"]["destination"]
        fill_request["interactCommand"]["verb"] = "FILL"
        return fill_request

    def _build_toggle_request(self, input_toggle_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_toggle_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_toggle_request["object"]["mask"], color_image_index)
            input_toggle_request["object"]["name"] = object_id
        toggle_request = self._get_interact_command_json()
        toggle_request["interactCommand"]["destinationObjectID"] = input_toggle_request["object"]["name"]
        toggle_request["interactCommand"]["verb"] = "TOGGLE"
        return toggle_request

    def _build_examine_request(self, input_examine_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_examine_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_examine_request["object"]["mask"], color_image_index)
            input_examine_request["object"]["name"] = object_id
        examine_request = self._get_interact_command_json()
        examine_request["interactCommand"]["destinationObjectID"] = input_examine_request["object"]["name"]
        examine_request["interactCommand"]["verb"] = "EXAMINE"
        return examine_request

    def _build_camera_change_request(self, input_camera_change_request):
        camera_change_request = self._get_camera_change_command_json()
        camera_change_request["camChangeCommand"]["mode"] = input_camera_change_request["mode"]
        return camera_change_request

    def _build_look_request(self, input_look_request):
        look_request = {}
        if input_look_request["direction"].lower() == "up":
            look_request["commandType"] = "LookUp"
            look_up_magnitude = RGActionsConstant.DEFAULT_LOOK_MAGNITUDE
            if "magnitude" in input_look_request:
                look_up_magnitude = input_look_request["magnitude"]
            look_request["panAndLookCommand"] = {"magnitude": look_up_magnitude}
        elif input_look_request["direction"].lower() == "down":
            look_request["commandType"] = "LookDown"
            look_down_magnitude = RGActionsConstant.DEFAULT_LOOK_MAGNITUDE
            if "magnitude" in input_look_request:
                look_down_magnitude = input_look_request["magnitude"]
            look_request["panAndLookCommand"] = {"magnitude": look_down_magnitude}
        elif input_look_request["direction"].lower() == "around":
            look_request["commandType"] = "LookAround"
            field_of_view_value = RGActionsConstant.DEFAULT_LOOK_AROUND_MAGNITUDE
            if "magnitude" in input_look_request:
                field_of_view_value = int(input_look_request["magnitude"])
            look_request["lookAroundCommand"] = {"fieldOfView": field_of_view_value}
            if "shouldRotate" in input_look_request:
                look_request["lookAroundCommand"]["shouldRotate"] = input_look_request["shouldRotate"]
            else:
                look_request["lookAroundCommand"]["shouldRotate"] = False
        return look_request

    def _build_highlight_request(self, input_highlight_request):
        if self.object_output_type == ObjectOutputType.OBJECT_MASK:
            color_image_index = input_highlight_request["object"].get("colorImageIndex", 0)
            object_id = self.find_object_id(input_highlight_request["object"]["mask"], color_image_index)
            input_highlight_request["object"]["name"] = object_id
        highlight_request = self._get_highlight_command_json()
        highlight_request["highlightCommand"]["instanceID"] = input_highlight_request["object"]["name"]
        if "shouldRotateToObject" in input_highlight_request["object"]:
            highlight_request["highlightCommand"]["shouldRotateToObject"] = input_highlight_request["object"]["shouldRotateToObject"]
        if "shouldRotateBack" in input_highlight_request["object"]:
            highlight_request["highlightCommand"]["shouldRotateBack"] = input_highlight_request["object"]["shouldRotateBack"]
        return highlight_request
