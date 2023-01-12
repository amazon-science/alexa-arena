# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from jinja2 import pass_context
import time
import datetime 
import os
import numpy as np
import re
import uuid
import json
from types import SimpleNamespace
import boto3
import random

import torch
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import cv2

from util.name_mapping import action_list, prepositions, room_names, readable_type_matching_dict, synonym_mapping, object_to_vision_class
from util.utils import compress_mask, process_image_for_model


def process_outputs(outputs_list: dict, object_class: str):
    """
    Processes the outputs of the CV model to return a sings matching instance mask prediction corresponding
    to the object instance ID.
    """
    ML_Toolbox_path = os.getenv('ML_TOOLBOX_DIR')
    class_to_idx_path = ML_Toolbox_path + '/modeling/inference/cv_model_utils/class_to_idx_3.json'
    with open(class_to_idx_path) as f:
        class2idx = json.load(f)
    idx2class = {v: k for k, v in class2idx.items()}
    predicted_class, predicted_score, predicted_mask = None, None, None

    score_threshold = 0.0
    img_idx = -1
    mask_sum = -1.0
    pixel_mask_prob_threshold = 0.75
    # Maps semantic parser model classes to vision model classes
    class_of_object_id = object_to_vision_class[object_class]
    for i, outputs in enumerate(outputs_list):
        pred_masks = outputs[0]["masks"].cpu()
        pred_labels = outputs[0]["labels"].cpu()
        scores = outputs[0]["scores"].cpu()
        for m in range(pred_masks.shape[0]):
            pred_score = scores[m].item()
            if pred_score > score_threshold:
                pred_label = idx2class[pred_labels[m].item()]
                if 'shelf' in pred_label.lower():
                    pred_label = 'Shelf'
                pred_mask = pred_masks[m][0].cpu().detach().numpy()
                pred_mask[pred_mask >= pixel_mask_prob_threshold] = 1
                pred_mask[pred_mask < pixel_mask_prob_threshold] = 0
                curr_mask_sum = np.sum(pred_mask)
                # if there are multiple instances of the object, we return the instance with the maximum area
                if (pred_label == class_of_object_id) and (curr_mask_sum  > mask_sum):
                    mask_sum = curr_mask_sum
                    predicted_class = pred_label
                    predicted_mask = pred_mask
                    predicted_score = pred_score
                    img_idx = i
    return predicted_class, predicted_score, predicted_mask, img_idx


def process_gt_mask(out, images_list, device, cpu_device, model=None):
    '''
    Gets object ID from out, maps the object ID to a class instance segmentation mask
    from the CV model and adds the binary mask  predicted by the model for that instance of the object.
    '''
    return_out = []
    outputs_list = []
    for image in images_list:
        image_input = process_image_for_model(image)
        image_input = list(img.to(device) for img in image_input)
        outputs = model.model(image_input)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs_list.append(outputs)

    for elt in out:
        if elt["type"].lower() != 'rotate' and "object" in elt[elt["type"].lower()] and not(elt["type"].lower() == 'goto' and "officeRoom" in elt['goto']["object"].keys()):
            if "name" in elt[elt["type"].lower()]["object"]:
                object_class = elt[elt["type"].lower()]["object"]["name"]
            elif "destination" in elt[elt["type"].lower()]["object"]:
                object_class = elt[elt["type"].lower()]["object"]["destination"]
            else:
                print("Object ID cannot be determined")
            pred_class, _, pred_mask, img_idx = process_outputs(outputs_list, object_class)
            # Case 1: if mask for the object id is not predicted by the model in the image
            if pred_class == None:
                return_out.append(elt)
            # Case 2: if mask is present
            else:
                mask = pred_mask
                compressed_mask = compress_mask(mask)
                # 2a) if mask is all empty
                if len(compressed_mask) == 0:
                    return_out.append(elt)  # Empty mask
                # 2b) if mask isnt empty
                else:
                    elt[elt["type"].lower()]["object"]["mask"] = compressed_mask
                    elt[elt["type"].lower()]["object"]["colorImageIndex"] = img_idx
                    return_out.append(elt)
        else: # If no object then no mask needed and hence copy as it is
            return_out.append(elt)
    return return_out



################################################################################
#Define utility functions
################################################################################

def find_matching_object_root(sentence, unique_object_list):
        min_index = np.inf  
        predicted_object = None   
        for object_root in unique_object_list: 
            if object_root not in synonym_mapping:
                synonym_mapping[object_root] = [object_root] #The object root is the only defined synonym of the word if it doesnt occur in the synonym mapping dictionary
            for object in synonym_mapping[object_root]: #Takes care of all synonyms of the base object 
                index = sentence.lower().find(object.lower() + ' ')   #See if the object string occurs in the middle of a sentence. returns -1 if not found and index of first occurence in string  if found
                if index == -1:
                    index = sentence.lower().find(object.lower() + '.') #Match to see if it occurs at the end of the sentence
                    
                if index != -1 and index < min_index:
                    min_index = index 
                    predicted_object = object_root  
                    
                #Dont break this loop and let this scan all the objects to allow for multiple objects 
                # in the sentence due to use of prepositions or relative object location.       
                # Presently returns the object which occurs first in the sentence
                # Another way to do this could be to chop off part after preposition or qualifiers (taken care of for use and place where we need to keep the receptacle)
        return predicted_object
        
        
def predict_action_and_object(instr, instr_hist='' ):

    sentence = instr 

    
    #Get a list of all unique human readable object types
    unique_object_list = list(set([ readable_type_matching_dict[key] for key in readable_type_matching_dict.keys()]))
   
    if ' and ' in sentence: #Compound sentence, needs to be broken up
        sentences = sentence.split(' and ')  
        sentences = [(sentence + ' .') for sentence in sentences ] #Adding a space and full stop to each part of the sentence
    else:
        sentences = [sentence] #To ensure same type 
        
    out = []
    for sentence in sentences: #To handle each sentence piece in the compound sentence
        sentence += ' ' #Add a space at the end to allow end of sentence matching
        print("Handling instruction: " + sentence)
        
        if "camera" in sentence:
            if "first" in sentence or "egocentric" in sentence: #Switch to egocentric view
                action_item_dict = {
                    "id": str(uuid.uuid1()), 
                    "type": "CameraChange", 
                    "camerachange": {
                        "mode": "Camera_TAM",
                    }
                }
            else: #Default to third person view
                action_item_dict = {
                    "id": str(uuid.uuid1()), 
                    "type": "CameraChange", 
                    "camerachange": {
                        "mode": "Camera_Regular",
                    }
                }
                
            out.append(action_item_dict)
            continue

        if "look around" in sentence:
            action_item_dict = {
                    "id": str(uuid.uuid1()), 
                    "type": "Look", 
                    "look": {
                        "direction": "Around",
                        "magnitude": 100
                    }
                }
            out.append(action_item_dict)
            continue
        
        
        predicted_action = None
        predicted_object = None #Use to check if a verb and object were found in sentence
        
        if "turn on" in sentence or "turn off" in sentence or "switch on" in sentence or "switch off" in sentence:
            predicted_action = "Toggle"
        elif "back " in sentence or " forward " in sentence or " back" in sentence:
            predicted_action = 'Move'
        else:
            for action in action_list:
                pattern = re.compile(action[0] + ' ', re.IGNORECASE) #Case insenstive matching. Space to ensure exact action verb match and not substrings
                if re.search(pattern, sentence):
                    print(f'The Action {action[0]} matches the string {sentence}') 
                    predicted_action = action[1] #RG compatible action
                    break #Break the search after first matching action found
                
        if predicted_action != None:
            print("Predicting action: " + predicted_action)
        else:
            print("No matching action found. Continuing with next instruction")
            continue    
        
        #Handle "PUT", "USE", "ROTATE", "MOVE" differently than other actions since they interact differently with the objects   
        if predicted_action == 'Move':
            if 'backward' in  sentence.lower() or 'back ' in  sentence.lower() or ' back' in  sentence.lower():
                #Move backward
                action_item_dict = {
                    "id": str(uuid.uuid1()), 
                    "type": "Move", 
                    "move": {
                        "direction": "Backward", # Forward or Backward
                        "magnitude": 1, # number of steps, each step is 1.0 magnitude in RG API
                    }
                }
                
            else:
                #Move forward (default)
                action_item_dict = {
                    "id": str(uuid.uuid1()), 
                    "type": "Move", 
                    "move": {
                        "direction": "Forward", # Forward or Backward
                        "magnitude": 1, # number of steps, each step is 1.0 magnitude in RG API
                    }
                }
            
            out.append(action_item_dict)
            continue
        
        if predicted_action == 'Rotate':
            if ' right ' in  sentence.lower() or ' light ' in  sentence.lower(): #light is a common ASR error for right
                #turn right 90 degrees
                action_item_dict = {
                    "id":  str(uuid.uuid1()),
                    "type": "Rotate", 
                    "rotation": {
                        "direction": "Right", # Right or Left
                        "magnitude": 45, 
                    }
                } 
            elif 'around' in sentence.lower():
                #turn right 180 degrees
                action_item_dict = {
                    "id":  str(uuid.uuid1()),
                    "type": "Rotate", 
                    "rotation": {
                        "direction": "Right", # Right or Left
                        "magnitude": 180, # number of steps, each step rotates 15 degrees, default is 90 degree rotation
                    }
                } 
            else:
                #turn left 90 degrees
                action_item_dict = {
                    "id":  str(uuid.uuid1()),
                    "type": "Rotate", 
                    "rotation": {
                        "direction": "Left", # Right or Left
                        "magnitude": 45, # number of steps, each step rotates 15 degrees, default is 90 degree rotation
                    }
                }     
            out.append(action_item_dict)
            continue

        if predicted_action == 'Look':
            if 'note' in sentence.lower() or 'hint' in sentence.lower() or 'sticky' in sentence.lower(): #To allow "look at the note" etc to work  
                predicted_action = 'Examine'
                
            elif 'down' in  sentence.lower() :
                action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": "Look",
                    "look": {
                        "direction": "Down",
                        "magnitude": 45.0,
                    }
                }
                out.append(action_item_dict)
                continue
            
            elif 'straight' in  sentence.lower() or 'ahead' in  sentence.lower():
                action_item_dict = {
                    "id": str(uuid.uuid1()), 
                    "type": "Move", 
                    "move": {
                        "direction": "Forward", # Forward or Backward
                        "magnitude": 0, # number of steps, each step is 1.0 magnitude in RG API
                    }
                } #Move forward wiht 0 magnitude resets the camera angle to horizontal
                out.append(action_item_dict)
                continue
            
            else:
                action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": "Look",
                    "look": {
                        "direction": "Up",
                        "magnitude": 45.0,
                    }
                }
                out.append(action_item_dict)
                continue
                    
        if predicted_action == 'Place': #No need to worry about the primary object in this, just identify the receptacle, since by default the object in hand will be kept
            #handle this separately 
            for preposition in prepositions:
                if preposition in sentence:
                    sentence = ' ' + sentence.split(preposition)[1] + ' ' #Keep only the part after the preposition to look for the secondary object there

        elif predicted_action == 'Throw' or predicted_action == 'Pour' or predicted_action == 'Fill' or predicted_action == 'Clean': #Handle the 2 arguement use separately
               
            if "throw" in sentence and ("dart" in sentence or "dark" in sentence or "dot" in sentence): #Handle throwing dartboard case separately to circumvent frequent ASR issues
                predicted_object1 = find_matching_object_root(" dart ", unique_object_list)
                predicted_object2 = find_matching_object_root(" dartboard ", unique_object_list)
                
                action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": predicted_action, 
                    predicted_action.lower(): {
                        "object": {
                        "colorImageIndex": 0,
                        "name": predicted_object2 #Source object is assumed to be object in hand, so only pass destination
                        }
                    }
                }
                out.append(action_item_dict )  
                continue 
            
                
            sentence1 = None
            sentence2 = sentence #If no preposition occur, then the whole sentence will be used to find object #2
            for preposition in prepositions:
                if preposition in sentence:
                    sentences = sentence.split(preposition)
                    sentence1 = ' ' + sentences[0] + ' ' 
                    sentence2 = ' ' + sentences[1] + ' '
                    break
            
            if sentence1 != None:    
                predicted_object1 = find_matching_object_root(sentence1, unique_object_list) #Matches the object to all possible object root using the synonyms of the object root and returns the first matched object (the one which occurs earlier in the sentence). Returns None if no object match is found
                if predicted_object1 != None:
                    print("Predicting object 1: " + predicted_object1) 
                else:
                    print("Predicting object: None") 
            else:
                predicted_object1 = None
            
            if sentence2 != None:        
                predicted_object2 = find_matching_object_root(sentence2, unique_object_list) #Matches the object to all possible object root using the synonyms of the object root and returns the first matched object (the one which occurs earlier in the sentence). Returns None if no object match is found
                if predicted_object2 != None:
                    print("Predicting object 2: " + predicted_object2) 
                else:
                    print("Predicting object: None") 
            else:
                predicted_object2 = None
                
            if predicted_object1 != None and predicted_object2 != None:
                if ' with ' in sentence: # For example "fill the bowl with milk" or "clean the bowl with cloth"
                    if predicted_action == 'Fill' or predicted_action == 'Pour': 
                        action_item_dict = {
                            "id": str(uuid.uuid1()),
                            "type": predicted_action, 
                            predicted_action.lower(): {
                                "object": {
                                "colorImageIndex": 0,
                                "name": predicted_object1  #Source to be filled is object of interest. Destination is usually the sink
                                }
                            }
                        }
                    elif predicted_action == 'Clean':
                        action_item_dict = {
                            "id": str(uuid.uuid1()),
                            "type": predicted_action, 
                            predicted_action.lower(): {
                                "object": {
                                "colorImageIndex": 0,
                                "name": predicted_object2  #object to be cleaned is usually assumed to be in hand. So return the target of where to clean it at
                                }
                            }
                        }
                    else:
                        raise NotImplementedError("Rule not implemented")

                elif ' to ' in sentence or ' into ' in sentence: # For example "pour milk to the bowl"
                    if predicted_action == 'Fill' or predicted_action == 'Pour': 
                        action_item_dict = {
                            "id": str(uuid.uuid1()),
                            "type": predicted_action, 
                            predicted_action.lower(): {
                                "object": {
                                "colorImageIndex": 0,
                                "name": predicted_object2  #Source to be filled is object of interest.
                                }
                            }
                        }
                    else:
                        raise NotImplementedError("Rule not implemented")

                else:
                    raise NotImplementedError("Rule not implemented")

                out.append(action_item_dict )  
                continue  
            
            elif predicted_object2 != None:   #Revert to TAM_1 as the source
                action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": predicted_action, 
                    predicted_action.lower(): {
                        "object": {
                        "colorImageIndex": 0,    
                        "name": predicted_object2 #Assume TAM as other object and only return the main interactable object
                        }
                    }
                }
                out.append(action_item_dict )     
                continue 
                
            else:
                print("No usable objects found for action: " + str(predicted_action))    
                continue 
        
        elif predicted_action == 'Goto': #Check if the goto action issued is for a room
            for room in room_names:
                if room[0] in sentence:
                    room_id = room[1]
                    action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": "Goto", 
                    "goto": {
                        "object": {
                        "officeRoom": room_id, #room name instead of object ID
                        }
                    }
                    }
                    out.append(action_item_dict)
                    continue  
        
        #Continue searching the object for all other action types (and also for place)      
        predicted_object = find_matching_object_root(sentence, unique_object_list) #Matches the object to all possible object root using the synonyms of the object root and returns the first matched object (the one which occurs earlier in the sentence). Returns None if no object match is found
        
        if predicted_object == 'stickynote' and predicted_action == 'Open': #To handle open the sticky note occurence
            predicted_action = 'Examine'
            
        elif predicted_object == 'freezeray' and ('monitor' in sentence.lower() or 'computer' in sentence.lower()) and (predicted_action == "Toggle" or predicted_action == 'Goto'): 
            predicted_object = "freezeraymonitor" #To allow goto and toggle of freeze ray computer instead of the freeze ray itself
        
        elif predicted_object == 'laser' and ('monitor' in sentence.lower() or 'computer' in sentence.lower()) and (predicted_action == "Toggle" or predicted_action == 'Goto'): 
            predicted_object = "lasermonitor" #To allow goto and toggle of laser computer instead of the freeze ray itself
        
        elif predicted_object == 'laser' and predicted_action == "Toggle": #To enable "turn on the freeze ray"
           predicted_object = "lasermonitor"
        
        elif predicted_object == 'freezeray' and predicted_action == "Toggle": #To enable "turn on the laser"
           predicted_object = "freezeraymonitor"
               
        elif predicted_action == 'Toggle' and predicted_object == None and 'power' in  sentence.lower(): #Turn on the power
            predicted_object = 'lever' #Fusebox lever
        
        elif predicted_action == 'Toggle' and predicted_object == 'fusebox': #Turn on the fusebox
            predicted_object = 'lever' #Fusebox lever    
            
        elif predicted_object == 'monitor':
            if 'freeze ray' in sentence.lower():
                predicted_object = "freezeraymonitor"
            elif 'laser' in sentence.lower():
                predicted_object = "lasermonitor"
                
        elif predicted_object == 'actionfigure' and 'cartridge' in sentence.lower():     
            predicted_object = 'actionfigurecartridge'
                            
        if predicted_object != None:
            print("Predicting object: " + predicted_object) 
        else:
            print("Predicting object: None") 
        #TODO: Handle the special case for fridge where upper and lower fridge doors need to be different for the open and close actions
        #Similar for kitchen cabinets and drawers

        if predicted_action != None and predicted_object != None and predicted_action != "Use" and predicted_action != "Break": 
            action_item_dict = {
                "id": str(uuid.uuid1()),
                "type": predicted_action, 
                predicted_action.lower(): {
                    "object": {
                    "colorImageIndex": 0,
                    "name": predicted_object, 
                    }
                }
            }
            out.append(action_item_dict ) 
            
        elif predicted_action == "Use" and predicted_object != None:
            if "yesterday machine" in sentence or "time machine" in sentence: #Handles a case like switch on the yesterday machine separately to avoid matching it to light switch
                predicted_object = 'yesterdaymachine'
                predicted_action = "Toggle"
            elif "switch" in sentence: #Handles a case like switch on the yesterday machine separately to avoid matching it to light switch
                predicted_object = 'switch'
                predicted_action = "Toggle"    
            
            if predicted_action != "Toggle" :    
                action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": predicted_action, 
                    predicted_action.lower(): {
                        "object": {
                        "colorImageIndex": 0,
                        "name": predicted_object #Assume TAM as other object and only return the main interactable object
                        }
                    }
                }
            else:
                action_item_dict = {
                    "id": str(uuid.uuid1()),
                    "type": predicted_action, 
                    predicted_action.lower(): {
                        "object": {
                        "colorImageIndex": 0,
                        "name": predicted_object
                        }
                    }
                }
            out.append(action_item_dict )             
            
        elif predicted_action == "Break" and predicted_object != None:
            action_item_dict = {
                "id": str(uuid.uuid1()),
                "type": predicted_action, 
                predicted_action.lower(): {
                    "object": {
                    "colorImageIndex": 0,
                    "name": predicted_object #Assume hammer in the hand to be the source and object to be broken as the destination. Return only the other interactable
                    }
                }
            }
            out.append(action_item_dict )              
            
        #TODO Handle special case where object is found but action isnt or vice verse or both
        #Possibly use BERT/POS+Word2Vec based matching then

    if len(out) == 0:
        #No matching action and/or object pair was found in the sentence
        print("Error identifying action-object pair from instruction")
    return out, instr_hist #List of dictionaries with each dictionary being in the action interface compatible format. Would have just 1 entry for a simple sentence and could have 2 or more for compound sentences like goto to fridge and pickup the carton of milk and the joint instruction history
