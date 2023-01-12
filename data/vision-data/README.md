This README dives into the vision data.

## Data Download

Running `scripts/fetch_vision_data.sh` downloads the zip files `data1.zip` and `data2.zip`  into the `data/vision_data` folder and the object_manifests into the `data` folder of this repository. Unzipping this file extracts all the data containing the images, ground truth segmentation masks and the associated metadata into multiple folders `object_detection_data_v<x>`.


## Folder Structure

This is the folder structure:

```
data/vision_data
    --`object_detection_data_v1`
        --images_run1_OfficeLayout1
        --images_run1_OfficeLayout2
            --train
            --<img_num>_color.png
                    --<img_num>_seg.png
                    --<img_num>_metadata_<anchor_object>.json
                    .
                    .
                    .
            --validation
            --<img_num>_color.png
                    --<img_num>_seg.png
                    --<img_num>_metadata_<anchor_object>.json
                    .
                    .
                    .
            --log file
        .
        .
        .
    --`object_detection_data_v2`
        --images_run1_OfficeLayout1
        --images_run1_OfficeLayout2
            --train
                --<img_num>_color.png
                --<img_num>_seg.png
                --<img_num>_metadata_<anchor_object>.json
                .
                .
                .
            --validation
                --<img_num>_color.png
                    --<img_num>_seg.png
                    --<img_num>_metadata_<anchor_object>.json
                    .
                    .
                    .
            --log file
        .
        .
        . 
```

## Overview

Inside the `object_detection_data_v<x>` folders, there are multiple further folders `images_run<run_num>_<layout>` which correspond to different runs and different game layouts used for that specific run in the data collection process.

Inside each of the run folders, there are the following:
1. Either a `train` or `validation` folder, which have the respective train and validation images and associated metadata respectively. Not every run will have a validation folder, we have randomly selected images across the entire dataset and have created a validation set.
2. There is a log file which has all the logs of the robot trajectory that was used to generate the images and the associated metadata. This informartion can be obtained in a more structured way from the metadata, but this is a more readable file that can used to take a quick look at the trajectory.

Inside either the `train` or `validation` folders contain the color images, ground truth segmentation images and corresponding metadata files. The files and the naming convention is as follows:
`--<img_num>_color.png`
`--<img_num>_seg.png`
`--<img_num>_metadata_<anchor_object>.json`

The color and segmentation images are the RGB egocentric views of the robot and the corresponding ground truth segmentation images respectively. 
The `anchor_object` in some of the metadata files is the object to which that specific image and correcponding segmentation image and metadata generation was anchored around. For example, the robot was instructed to go to the bowl move back, rotate 30 degrees, move forward and then an image was captured. In this example, the bowl is the `anchor_object`. A caveat to note is that since the simulation environment has other objects and the robot can move around in this random walk, it is not necessary that all the images anchored around an object actually have the object in the image since the object may have been occluded during this process. However, this can easily be verified using the rgb mapping for each object in the metadata (if the image actually has the object, its segmentation would be present in the segmentation image). See the metadata section below for more details.

## Metadata

We now delve into the metadata file. 
The image and corresponding ground truth segmentation image can be obtained from the `image_annotations` field in the metadata file. This is a list of object annotations for the image. Each annotation has 4 fields ['rgb', 'object_id', 'bbox', 'object_type'].
```
medata = {
    .
    .
    "image_annotations": [
        {
            'rgb': This is the corresponding rgb value in the segmentation image for that object annotation
            'object_id': the object instance id
            'bbox': the bounding box coordinates. the coordinates are
                    [
                        rmin (top left row coordinate),
                        cmin (top left column coordinate),
                        rmax (bottom right row coordinate),
                        cmax (bottom right column coordinate)
                    ].
            'object_type': The general class of that object instance, for eg, all monitor instances are mapped to computer, etc.
        },
        .
        .
    ]
    .
    .

}
```

The metadata also contains information about all the commands executed prior to the the capture of that image. This is provided as a list in the `command` field.

The `object_id` field gives the object that the robot navigation was anchored to create that image. However, as previously mentioned, taking the random walk around the object might have occluded this original anchor object. This can be easily verified using the rgb mapping in the `image_annotations` field for each annotation. If the `object_id` field is None, then the image was not collected trying to anchor movements around any specific object (e.g. it could have been executing some mission)

```
metadata = {
    .
    .
    "command": [
        command1,
        command2, 
        command3,
        .
        .
    ]
    .
    .
}
```

The `sceneMetadata` field under the `response` field has other information about the image. The `objects` field in the `response` field can be accessed to extract state information of all objects in the layout, which can then be processed to extract the state objects present in the image/scene by using this information in conjunction with the `image_annotations` and `object_id` information. The depth images in the vision dataset are not usable at the moment and will be fixed with an update.

#### To read base64 string encoded images
The images in the responses are encoded using base64. You can extract and save the image to disk as a png file using the following method. 
```
def read_string_image(string_image, path_to_save):
    encoded_image = str.encode(string_image)
    decoded_image = base64.decodebytes(encoded_image)
    with open(path_to_save, "wb") as file_handler:
        file_handler.write(decoded_image)
```

To convert the image to an array that can be used for processing:

```
import numpy as np
def get_image_array(decoded_image):
    image_buffer = np.asarray(bytearray(decoded_image), dtype="uint8")
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_array
```
