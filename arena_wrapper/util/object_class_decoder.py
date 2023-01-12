# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import numpy as np

from arena_wrapper import AppConfig

readable_type_matching_dict = {
    "Fork_01": "fork",
    "CoffeeMug_Yellow": "coffeemug",
    "CoffeeMug_Yellow_ContainsCoffee": "coffeemug",  # note that this is substring of broken coffeemug
    'CoffeeMug_Boss': "coffeemug",
    'CoffeeMug_Boss_ContainsCoffee': "coffeemug",
    'CoffeeCup_Open_Empty_01': "coffeemug",
    'CoffeeCup_Open_Empty_02': "coffeemug",
    'CoffeeMug_Yellow_Broken': "brokencoffeemug",
    'CoffeeMug_Boss_Broken': "brokencoffeemug",
    "CoffeePotEmpty_01": "coffeepot",
    "CoffeePot_01": "coffeepot",
    'CoffeePot_WithCoffee_01': "coffeepot",
    "Jar_PeanutButter_01": "jar",
    'Jar_PeanutButter_Eaten_01': "jar",
    'Jar_Jam_Eaten_01': "jar",
    'CandyJar_01': 'jar',
    'Jar_Jam_01': 'jar',
    'CanSodaNew_01': 'sodacan',
    'CanSoda_01': 'sodacan',
    'CanSodaNew_Open_01': 'sodacan',
    'CanSodaNew_Crushed_01': 'sodacan',
    "BreadSlice_01": "bread",
    "CoffeeBeans_01": "coffeebeans",
    "PaperCup_Crushed_01": "papercup",
    "Knife_01": "knife",
    "MilkCarton_01": "milkcarton",
    "SandwichHalf_01": "sandwich",
    'SandwichHalfEaten_01': "sandwich",
    "FoodPlate_01": "plate",
    'FoodPlateDirty_01': 'plate',
    'FoodPlateBroken_01': 'plate',
    "Bowl_01_Broken": "brokenbowl",  # Check whether to just map this to bowl
    'Bowl_ContainsMilk_01': 'bowl',
    # note that this is substring of brokenbowl and broken bowl needs to be matched first
    'Bowl_ContainsCereal_01': 'bowl',
    'Bowl_ContainsMilkAndCereal': 'bowl',
    'Bowl_ContainsCoffee': 'bowl',
    'Bowl_01': 'bowl',
    "CakeSlice_02": "cake",
    'CakeCut_02': "cake",
    'SM_Prop_Table_02': 'table',
    'Table_Metal_01': 'table',
    'TableRound_02': 'table',
    'TableRoundSmall_02': 'table',
    'PaperCup_01': "papercup",
    'CandyBar_01': 'candybar',
    'CandyBar_Open_01': 'candybar',
    'CandyBar_Eaten_01': 'candybar',
    'Shelf_01': 'shelf',
    'AP_Prop_Shelf_Wall_04': "shelf",
    'AP_Prop_Shelf_06': "shelf",
    'Shelves_Tall_01': "shelf",
    'AP_Prop_Shelf_Wall_Laser': "shelf",
    'AP_Prop_Shelf_Wall_FreezeRay': "shelf",
    'Bookshelf_Wooden_01': "bookshelf",  # Figure out how to distinguish this from sb phrase shelf
    'KitchenCabinet_02': 'cabinet',  # Notice that shelf and cabinet are different
    'KitchenCabinet_01': 'cabinet',
    'KitchenCabinet_01_Trapped': 'cabinet',
    'KitchenCounterDrawer_03': 'drawer',  # Keeping this separate from shelf/cabinet
    'KitchenCounterDrawer_02': 'drawer',  # Keeping this separate from shelf/cabinet
    'ReceptionDesk': 'desk',  # Different from table
    'Desk_01': 'desk',
    'ManagerDesk': 'desk',
    'AP_Prop_Desk_Red': 'desk',
    'AP_Prop_Desk_Green': 'desk',
    'AP_Prop_Desk_Blue': 'desk',
    'AP_Prop_Desk_Yellow': 'desk',
    'Computer_Monitor_New': "monitor",
    'Computer_Monitor_01': "monitor",
    'V_Monitor_Laser': 'lasermonitor',
    'V_Monitor_Gravity': "monitor",
    'V_Monitor_Embiggenator': 'monitor',
    'V_Monitor_FreezeRay': 'freezeraymonitor',
    'V_Monitor_Portal': 'monitor',
    'Computer_Monitor_Broken': 'monitor',
    'Cake_02': 'cake',
    'Pear_01': 'pear',
    'Pear_01_Eaten': 'pear',
    'GravityPad': 'gravitypad',
    'Hammer': 'hammer',
    'Burger_04': 'burger',
    'BreadLoaf': 'bread',
    'ColorChanger_Button_Blue': "bluecolorchangerbutton",
    'ColorChanger_Button_Green': "greencolorchangerbutton",
    'ColorChanger_Button_Red': "redcolorchangerbutton",
    'VendingMachine_01_E5_Button': 'vendingmachinebutton',  # See how to differentiate these if needed
    'VendingMachine_01_E7_Button': 'vendingmachinebutton',
    'VendingMachine_01_B4_Button': 'vendingmachinebutton',
    'VendingMachine_01_M8_Button': 'vendingmachinebutton',
    'VendingMachine_01': 'vendingmachine',
    'BurgerEaten_04': 'burger',
    'TrashCan_01': 'trashcan',
    'Toast_01': 'toast',
    'Toast_02': 'toast',
    'Toast_03': 'toast',
    'Toast_04': 'toast',
    'Toast_04_Jam': 'toast',
    'Toast_04_PBJ': 'toast',
    'Toast_04_Jam_Eaten': 'toast',
    'Toast_04_PBJ_Eaten': 'toast',
    'Toast_04_Eaten': 'toast',
    'Cereal_Box_01': 'cerealbox',
    'YesterdayMachine_01': 'yesterdaymachine',
    'Microwave_01': 'microwave',
    'FridgeUpper_02': 'fridgeupper',
    'FridgeLower_02': 'fridgelower',
    'WallClock_01': 'clock',
    'Apple': 'apple',
    'AppleSlice_01': 'apple',
    'AppleCut_01': 'apple',
    'Apple_Eaten': 'apple',
    'DartBoard': 'dartboard',
    'BreadLoaf_Sliced': 'bread',
    'Dart': 'dart',
    'Laser': 'laser',
    'Laser_Tip': 'lasertip',
    'Radio_01': 'radio',
    'TeslaCoil': 'teslacoil',
    'Door_01': 'door',
    'Spoon_01': 'spoon',
    'Banana_01': 'banana',
    'BananaBunch_01': 'banana',
    'Banana_Peeled_01': "banana",
    'Banana_Eaten_01': "banana",
    'Trophy01': 'trophy',
    'FireAlarm_01': 'firealarm',
    'LightSwitch_01': 'switch',
    'Floppy_AntiVirus': 'floppy',
    'WaterCooler_01': 'watercooler',
    'Toaster_02': 'toaster',
    'PortalGenerator': 'portalgenerator',
    'Record_01': 'record',
    'Record_01_Broken': 'record',
    'ForkLift': 'forklift',
    'RoboticArm_01': 'roboticarm',
    'CoffeeMaker_01': 'coffeemaker',
    'ColorChangerStation': 'colorchangerstation',
    'TAMPrototypeHead_01': 'tamhead',
    'EAC_Machine': 'eacmachine',
    'MissionItemHolder': 'missionitemholder',
    'KitchenStool_01': 'stool',
    'WaterPuddle_01': 'puddle',
    'FuseBox_02': 'fusebox',
    'FuseBox_01': 'fusebox',
    'FuseBox_01_Lever': 'lever',
    'PackingBox': 'box',
    'KitchenCounterBase_03': 'counter',
    'KitchenCounterBase_02': 'counter',
    'KitchenCounterTop_02': 'counter',
    'CounterBase_03': 'counter',
    'KitchenCounter01': 'counter',
    'CoffeeUnMaker_01': 'unmaker',
    'FulllPaperTray_01': 'tray',
    'EmptyPaperTray': 'tray',
    'KitchenCounterSink_01': 'sink',
    'Handsaw': 'handsaw',
    'Screwdriver': 'screwdriver',
    'FreezeRay': 'freezeray',
    'Whiteboard_CoffeeUnmaker': 'whiteboard',
    'Whiteboard_YesterdayMachine': 'whiteboard',
    'WhiteBoard_01': 'whiteboard',
    'Broken_Cord_01': ' cord',  # Space needed to differentiate from record
    'Printer_3D': 'printer',
    'Embiggenator': 'embiggenator',
    'Floppy_Virus': 'virus',
    'WarningSign_01': 'warningsign',
    'Fork_Lift': 'forklift',
    'Carrot_01': 'carrot',
    'Carrot_Eaten_01': 'carrot',
    'PowerOutlet_01': 'poweroutlet',
    'Laser_CircuitBoard': 'circuitboard',
    'PinBoard_01': 'pinboard',
    'PinBoard_02': 'pinboard',
    'FireExtinguisher_01': 'extinguisher',
    'PieFruit_01': 'fruitpie',
    'PieFruitSlice_01': 'fruitpie',
    'PieFruitCut_01': 'fruitpie',
    'SafetyBarrier_02': 'safetybarrier',
    'DeskFan_Broken_01': 'fan',
    'DeskFan_New_01': 'fan',
    'CoffeeCup_Lid_01': 'lid',
    'CableFrayed_01': 'cable',
    'Printer_Cartridge': 'cartridge',
    'Donut_01': 'donut',
    'Donut_Eaten_01': 'donut',
    'StickyNote': 'stickynote',
    'Security_Button': 'securitybutton',
    'AutoChompers': 'chompers',
    'Laser_ControlPanel': 'controlpanel',
    # MS6 objects
    'Office_Chair': 'chair',
    'Manager_Chair': 'chair',
    'Keyboard': 'keyboard',
    'Printer_Cartridge_Lever': 'lever',
    'ActionFigure': 'actionfigure',
    'Cutting_Board': 'cutting board',
    'PBJ_Sandwich': 'sandwich',
    'TeslaCoil_Small': 'tesla coil',
    'Printer_Cartridge_Mug': 'mug',
    'Printer_Cartridge_Figure': 'actionfigurecartridge',
    'Floppy_AntiVirus_Broken': 'broken floppy',
    'Floppy_Virus_Broken': 'broken floppy',
    'Printer_Cartridge_Hammer': 'hammer',
    'Warehouse_Boxes': 'warehouse boxes',
    'Radio_01_Broken': 'broken radio',
    'LaserBase_toy': 'laser toy',
    # Some not used ones are mapped to a temporary string
    'SM_Bld_Wall_Window_Blinds_Open_04': 'notused',
    'SM_Prop_FlatPackCardboardBoxes_03': 'notused',
    'SK_Veh_Pickup_01_ToolBox': 'notused',
    'SM_Prop_Paper_Pile_01': 'notused',
    'AP_Prop_Lab_Tank_02': 'notused',
    'AP_Prop_Note_05': 'notused',
    'sign_short_caution_electrical': 'notused',
    'sign_tall_caution_carrot': 'notused',
    'sign_short_quantum_1': 'notused',
    'sign_tall_poster_tam_2': 'notused',
    'SM_Prop_PalletStack_02': 'notused',
    'SM_Prop_Book_Group_01': 'notused',
    'sign_diamond_carrot': 'notused',
    'AP_Prop_Minigolf_Club_01': 'notused',
    'SM_Prop_Paper_Pile_03': 'notused',
    'SM_Prop_Book_Group_07': 'notused',
    'SM_Tool_Drill_Chuck_01': 'notused',
    'SM_Prop_Book_Group_06': 'notused',
    'AP_Prop_Cabinets_01': 'notused',
    'SM_Prop_FolderTray_01': 'notused',
    'sign_short_caution_gravity_2': 'notused',
    'SM_Prop_Book_Group_05': 'notused',
    'SM_Prop_FlatPackCardboardBoxes_04': 'notused',
    'SM_Prop_FolderTray_04': 'notused',
    'SM_Bld_Wall_Metal_Slide_02': 'notused',
    'SM_Bld_Door_02': 'notused',
    'sign_short_poster_delwan_2': 'notused',
    'SM_Prop_Drink_Dispenser_01': 'notused',
    'SM_Prop_Paper_06': 'notused',
    'SM_Prop_Folder_PVC_01': 'notused',
    'AP_Prop_CorkBoard_02': 'notused',
    'SM_Prop_Warehouse_Light_04': 'notused',
    'sign_short_breakroom_2': 'notused',
    'SM_Prop_Buttons_05': 'notused',
    'SM_Prop_Folder_Holder_02': 'notused',
    'sign_short_warehouse_1': 'notused',
    'AP_Prop_Barrel_Water_01': 'notused',
    'AP_Prop_Folder_PVC_02': 'notused',
    'SM_Prop_Server_Node_01': 'notused',
    'SM_Prop_NetCable_03': 'notused',
    'SM_Prop_Book_Group_08': 'notused',
    'AP_Prop_Couch_06': 'notused',
    'sign_tall_caution_shrink': 'notused',
    'AP_Prop_Barrel_Open_01': 'notused',
    'SM_Prop_NotePad_01': 'notused',
    'SM_Prop_Book_Phone_Open_01': 'notused',
    'sign_tall_caution_freeze': 'notused',
    'sign_short_caution_restricted_1': 'notused',
    'SM_Item_Clipboard_01': 'notused',
    'SM_Prop_Cart_01': 'notused',
    'AP_Prop_Lab_Tank_01': 'notused',
    'sign_diamond_gravity': 'notused',
    'SM_Prop_Book_Group_02': 'notused',
    'SM_Prop_Book_Magazine_01': 'notused',
    'AP_Prop_Lab_MachinePanel_01': 'notused',
    'sign_short_caution_gravity_1': 'notused',
    'SM_Prop_Oxygen_Tank': 'notused',
    'AP_Prop_Fire_Extinguisher_01': 'notused',
    'SM_Prop_Folder_Holder_04': 'notused',
    'SM_Prop_FolderTray_03': 'notused',
    'AP_Prop_Plant_09': 'notused',
    'SM_Prop_Folder_PVC_02': 'notused',
    'SM_Prop_Lighting_Cable_Bulb_01': 'notused',
    'AP_Prop_Pen_01': 'notused',
    'SM_Prop_Wirespool_01': 'notused',
    'SM_Prop_Warehouse_Boxes_Stacked_04': 'notused',
    'sign_diamond_laser': 'notused',
    'sign_short_poster_delwan_1': 'notused',
    'SM_Prop_Book_Group_04': 'notused',
    'SM_Prop_Paper_04': 'notused',
    'SM_Prop_Server_Cabinet_01': 'notused',
    'sign_short_office_1': 'notused',
    'SM_Prop_AirVent_Wall_01': 'notused',
    'AP_Prop_Photocopier_01': 'notused',
    'SM_Prop_Certificate_01': 'notused',
    'SM_Prop_Wirespool_Small_01': 'notused',
    'AP_Prop_Safety_Barrier_02': 'notused',
    'sign_short_caution_shrink': 'notused',
    'sign_short_caution_quantum_2': 'notused',
    'SM_Prop_AirVent_01': 'notused',
    'AP_Prop_Pen_06': 'notused',
    'SM_Prop_PowerBoxes_01': 'notused',
    'sign_diamond_freeze': 'notused',
    'SM_Prop_Folder_Holder_01': 'notused',
    'AP_Prop_Bucket_02': 'notused',
    'AP_Prop_CardboardBox_Open_05': 'notused',
    'AP_Prop_Lab_Clamp_02_Arm_01': 'notused',
    'sign_tall_caution_electrical': 'notused',
    'sign_tall_poster_tam_1': 'notused',
    'SM_Prop_Folder_Holder_03': 'notused',
    'SM_Prop_Book_Group_03': 'notused',
    'SM_Prop_Folder_Manila_04': 'notused',
    'AP_Prop_Plant_01': 'notused',
    'Laser_Tip_Broken': 'notused',
    'AP_Prop_Lab_MachinePanel_02': 'notused',
    'sign_diamond_shrink': 'notused',
    'SM_Prop_Warehouse_Boxes_Stacked_03': 'notused',
    'sign_square_breakroom': 'notused',
    'SM_Prop_Powercable_02': 'notused',
    'AP_Prop_CardboardBox_Stack_02': 'notused',
    'SM_Tool_Buffer_01_Battery': 'notused',
    'SM_Prop_Calender_01': 'notused',
    'AP_Item_Tape_01': 'notused',
    'SM_Prop_Oxygen_Tank_Large': 'notused',
    'SM_Prop_Powercable_01': 'notused',
    'AP_Prop_Couch_02': 'notused',
    'SM_Prop_Papers_01': 'notused',
    'SM_Prop_Crate_Stack_01': 'notused',
    'SM_Prop_Plastic_Pipe_Spool_01': 'notused',
    'AP_Bld_Wall_Glass_Large_Door_01': 'notused',
    'sign_short_quantum_2': 'notused',
    'sign_diamond_fire': 'notused',
    'sign_short_robotics_1': 'notused',
    'SM_Prop_Powercable_03': 'notused',
    'SM_Prop_Folder_Manila_01': 'notused',
    'AP_Prop_Cellotape_01': 'notused',
    'sign_short_poster_delwan_4': 'notused',
    'SM_Tool_Handsaw_01': 'notused',
    'SM_Prop_Buttons_02': 'notused',
    'AP_Prop_Bin_Rubbish_01': 'notused',
    'SM_Prop_Scales_01': 'notused',
    'SM_Sign_Exit_02': 'notused',
    'sign_short_robotics_2': 'notused',
    'SM_Prop_Paper_05': 'notused',
    'SM_Prop_Warehouse_Platform_Trolley_01': 'notused',
    'sign_tall_caution_robotics': 'notused',
    'sign_short_poster_delwan_3': 'notused',
    'AP_Prop_Pen_03': 'notused',
    'AP_Item_Tool_Board': 'notused',
    'AP_Prop_PaperTray_01_Full_01': 'notused',
    'AP_Prop_Generator_Large_02': 'notused',
    'AP_Bld_Ceiling_Aircon_01': 'notused',
    'sign_tall_caution_laser': 'notused',
    'sign_short_breakroom_1': 'notused',
    'SM_Prop_Folder_Manila_02': 'notused',
    'AP_Prop_Print_Tube_01': 'notused',
    'Lab_Terminal': 'notused',
    'Deembiggenator_Crates': 'notused'
}


def parse_metadata(metadata, readable_type_matching_dict):
    # Used to parse the metadata to get objects, their locations, their states, their affordances, "objectType" and
    # their ID's. Also returns the location of the bot itself Takes as input the metadata object returned by the
    # arena environment and the readable_type_matching_dict which maps each RG defined object type to a human-readable
    # object type (this could be statically defined beforehand since it remains fixed for the entire task.)
    # Returns a dictionary of list of dictionaries where the top level keys are human-readable object types with each
    # of these human object types being mapped to all objects in the scene Each of these keys has an associated list
    # as value where list contains information of all the objects in the RG environment corresponding to that
    # specific object type category. The dictionary elements in this list have the same structure as the metadata
    # dictionary object corresponding to th various objects. Also returns the location of the bot itself
    if metadata is None:
        return None, None
    data = metadata["objects"]  # List of dictionaries of all objects including the bot itself
    unique_object_type_dict = {}
    bot_position = None
    for elt in data:
        if "TAM_" in elt["objectID"]:  # This is the bot
            # Dictionary with x, y and z as keys. Note, the (x,z) location is the coordinate
            # location and y is the height in unity
            bot_position = elt["position"]

        else:
            if elt["objectType"] not in readable_type_matching_dict:
                continue
            readable_object_type = readable_type_matching_dict[
                elt["objectType"]]  # Dictionary used to map the auxillary RG types to human readable ones
            if readable_object_type in unique_object_type_dict:
                # unique_object_type_dict[objectType].append({"objectID": elt["objectID"], "objectType": elt[
                # "objectType"], "position": elt["position"], "supportedVerbs": elt["supportedVerbs"],
                # "supportedStates": elt["supportedStates"], "currentStates": elt["currentStates"],
                # "parentReceptacle": elt["parentReceptacle"]}) #Retain selective entries from the main metadata
                # dictionary

                # Add the whole metadata entry for that object type to its list corresponding to its
                # redable name
                unique_object_type_dict[readable_object_type].append(elt)
            else:
                # Create a new list corresponding to this human readable object type
                unique_object_type_dict[readable_object_type] = [elt]
            # TODO: Will need to handle human readable synonyms to object type too
    return unique_object_type_dict, bot_position


def parse_metadata_only_bot_position(metadata, readable_type_matching_dict):
    # Use this if using preprocessed unique_object_type_dict
    data = metadata["objects"]  # List of dictionaries of all objects including the bot itself
    for elt in data:
        if "TAM_" in elt["objectID"]:  # This is the bot
            # Dictionary with x, y and z as keys. Note, the (x,z) location is the coordinate location and y is
            # the height in unity
            bot_position = elt["position"]
            break  # Bot position found, can break
        else:
            continue

    return bot_position


def locate_tams_room(metadata):
    # Use this to return the room TAM is in
    data = metadata["objects"]  # List of dictionaries of all objects including the bot itself
    bot_room = None
    for elt in data:
        if "TAM_" in elt["objectID"]:  # This is the bot
            # Dictionary with x, y and z as keys. Note, the (x,z) location is the coordinate
            # location and y is the height in unity
            bot_room = elt['currentRoom']
            break  # Bot position found, can break
        else:
            continue
    return bot_room


def find_matching_object(predicted_object, object_type_dictionary, bot_location, best_object_match_index=0):
    if best_object_match_index != 0:  # Handle case for second best separately
        return find_next_best_matching_object(predicted_object, object_type_dictionary, bot_location,
                                              best_object_match_index)

        # Predicted object is the human readable object type predicted
    # object_type_dictionary is a dictionary with keys as the human readable object types and value as a list of
    # dictionaries containing all instances of that object type This is used to find the correct object id based on
    # nearest distance from the bot and return its object ID returns the object_id for the best matched object
    if predicted_object not in object_type_dictionary:
        if AppConfig.runtime_platform == "Mac":
            return predicted_object
        else:
            return None
    matched_object_list = object_type_dictionary[predicted_object]
    min_dist = np.inf
    best_object_id = None
    for obj in matched_object_list:
        object_location = obj["position"]
        dist = (object_location['x'] - bot_location['x']) ** 2 + (object_location['z'] - bot_location['z']) ** 2
        if dist < min_dist:
            min_dist = dist
            best_object_id = obj["objectID"]
    return best_object_id


def find_next_best_matching_object(predicted_object, object_type_dictionary, bot_location, best_object_match_index):
    # Finds and returns the SECOND best object closest to the bot
    matched_object_list = object_type_dictionary[predicted_object]
    min_dist = np.inf
    second_min_dist = np.inf
    best_object_id = None
    second_object_id = None

    for object in matched_object_list:
        object_location = object["position"]
        dist = (object_location['x'] - bot_location['x']) ** 2 + (object_location['z'] - bot_location['z']) ** 2
        if dist < min_dist:
            second_min_dist = min_dist
            min_dist = dist
            second_object_id = best_object_id
            best_object_id = object["objectID"]

        elif dist < second_min_dist:
            second_min_dist = dist
            second_object_id = object["objectID"]

    return second_object_id


def find_object_ids_from_type(needed_object_types, metadata):
    needed_object_ids = []
    data = metadata["objects"]  # List of dictionaries of all objects including the bot itself
    for object_type in needed_object_types:
        for elt in data:
            if object_type == elt["objectType"]:
                needed_object_ids.append(elt["objectID"])
                break
            else:
                continue

    return needed_object_ids


# Will be used to get all computers of the type
def find_all_object_ids_from_type(metadata, needed_object_type='Computer_Monitor_01'):
    needed_object_ids = []
    data = metadata["objects"]  # List of dictionaries of all objects including the bot itself
    for elt in data:
        if needed_object_type == elt["objectType"]:
            needed_object_ids.append(elt["objectID"])  # Keep running to get all object IDs of this type
        else:
            continue

    return needed_object_ids

def identify_correct_shelf(commands, metadata, instr, predicted_object, unique_object_type_dict, bot_position, best_object_match_index, i):
    if "second" in instr or "two" in instr:  # Special case needed to support freeze ray mission
        predicted_object_id = find_all_object_ids_from_type( metadata, needed_object_type = 'AP_Prop_Shelf_Wall_04' )[0]
    elif "first" in instr or "one" in instr: # Special case needed to support Laser ray mission
        predicted_object_id = find_all_object_ids_from_type( metadata, needed_object_type = "AP_Prop_Shelf_Wall_Laser" )[0]
    else:
        # This is used to find the correct object id based on nearest distance from the bot and return
        # its object ID
        predicted_object_id = find_matching_object(predicted_object, unique_object_type_dict,
                                                   bot_position, best_object_match_index)
    return predicted_object_id

def convert_object_class_to_id(commands, metadata, instr=None):
    '''
    #Takes in the response object returned by action inference service and converts object class names to object IDs using the metadata
    #instr (NLP command) needed to implement certain rules
    '''

    # Used to parse the metadata to get objects, their locations, their states, their affordances, "objectType" and
    # their ID's. This is returned as a dictionary with keys as human readable object types and values as list of all
    # the objects in RG environemnt corresponding to that object type. The object specific information is encoded as
    # a dictionary with same structure as the metadata. Also returns the location of the bot itself
    unique_object_type_dict, bot_position = parse_metadata(metadata, readable_type_matching_dict)

    for i in range(len(commands)):
        command = commands[i]
        best_object_match_index = 0  # Default. Used to support second best matches

        if command["type"] == 'Dialog' or command["type"] == 'CameraChange' or \
                command["type"] == 'Move' or command["type"] == 'Rotate' or command["type"] == 'Look' or \
                ("object" not in command[command["type"].lower()]):
            continue  # No object here, can skip
        else:
            objects = command[command["type"].lower()]["object"]
            if "officeRoom" in objects:  # Room navigation commands don't need object-id
                continue
            if "name" in objects:
                predicted_object = objects["name"]
                if predicted_object is not None and predicted_object == "monitor" and \
                        locate_tams_room(metadata) == "MainOffice" and command["type"] == 'Goto':
                    computers = find_all_object_ids_from_type(metadata, needed_object_type='Computer_Monitor_01')
                    # This makes sure that they are always in same order ensuring consistency of first, second and third
                    computers.sort()
                    if 'first' in instr or 'left' in instr:
                        predicted_object_id = computers[0]
                    elif 'second' in instr or 'middle' in instr or 'center' in instr:
                        predicted_object_id = computers[1]
                    elif 'third' in instr or 'right' in instr:
                        predicted_object_id = computers[2]
                    elif 'one' in instr:  # Putting this separately as user might say "second one"
                        predicted_object_id = computers[0]
                    elif 'two' in instr:
                        predicted_object_id = computers[1]
                    elif 'three' in instr:
                        predicted_object_id = computers[2]
                    else:
                        # This is used to find the correct object id based on nearest distance from the bot and return
                        # its object ID
                        predicted_object_id = find_matching_object(predicted_object, unique_object_type_dict,
                                                                   bot_position, best_object_match_index)
                    commands[i][commands[i]["type"].lower()]["object"]["name"] = predicted_object_id
                elif predicted_object == "shelf" and locate_tams_room(metadata) == "Lab1":
                    predicted_object_id = identify_correct_shelf(commands, metadata, instr, predicted_object, unique_object_type_dict, bot_position, best_object_match_index, i)
                    commands[i][commands[i]["type"].lower()]["object"]["name"] = predicted_object_id
                elif predicted_object == 'vendingmachinebutton' and "button" in instr:
                    # Always press this button if a vending machine button is being pressed
                    predicted_object_id = "VendingMachine_01_E5_Button_10000"
                    commands[i][commands[i]["type"].lower()]["object"]["name"] = predicted_object_id
                elif command["type"] == 'Goto' and predicted_object == "monitor" \
                        and 'next' in instr:  # To enable "Goto the next computer"
                    # This will be used to tell the "find_matching_object_root" function to find the next best match
                    # besides the present one
                    best_object_match_index = 1
                    # This is used to find the correct object id based on nearest distance from the bot and return
                    # its object ID
                    predicted_object_id = find_matching_object(predicted_object, unique_object_type_dict,
                                                               bot_position, best_object_match_index)
                    commands[i][commands[i]["type"].lower()]["object"]["name"] = predicted_object_id
                elif predicted_object is not None:
                    # This is used to find the correct object id based on nearest distance from the bot and return
                    # its object ID
                    predicted_object_id = find_matching_object(predicted_object, unique_object_type_dict,
                                                               bot_position, best_object_match_index)
                    commands[i][commands[i]["type"].lower()]["object"]["name"] = predicted_object_id
                else:
                    print("Predicting object: None")
            if "source" in objects:
                predicted_object = objects["source"]
                if predicted_object == "shelf" and  locate_tams_room(metadata) == "Lab1":
                    predicted_object_id = identify_correct_shelf(commands, metadata, instr, predicted_object, unique_object_type_dict, bot_position, best_object_match_index, i)
                    commands[i][commands[i]["type"].lower()]["object"]["source"] = predicted_object_id
                elif predicted_object is not None and predicted_object in unique_object_type_dict and \
                        predicted_object != "TAM_1":
                    # This is used to find the correct object id based on nearest distance from the bot and return
                    # its object ID
                    predicted_object_id = find_matching_object(predicted_object, unique_object_type_dict,
                                                               bot_position, best_object_match_index)
                    commands[i][commands[i]["type"].lower()]["object"]["source"] = predicted_object_id
                else:
                    print("Keeping original source")
            if "destination" in objects:
                predicted_object = objects["destination"]
                if predicted_object == "shelf" and  locate_tams_room(metadata) == "Lab1":
                    predicted_object_id = identify_correct_shelf(commands, metadata, instr, predicted_object, unique_object_type_dict, bot_position, best_object_match_index, i)
                    commands[i][commands[i]["type"].lower()]["object"]["destination"] = predicted_object_id
                elif predicted_object is not None and predicted_object in unique_object_type_dict and \
                        predicted_object != "TAM_1":
                    # This is used to find the correct object id based on nearest distance from the bot and return
                    # its object ID
                    predicted_object_id = find_matching_object(predicted_object, unique_object_type_dict, bot_position,
                                                               best_object_match_index)
                    commands[i][commands[i]["type"].lower()]["object"]["destination"] = predicted_object_id
                else:
                    print("Keeping original destination")
    return commands
