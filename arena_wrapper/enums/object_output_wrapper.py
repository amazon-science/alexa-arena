# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from enum import Enum


# creating enumerations for object output types
class ObjectOutputType(str, Enum):
    OBJECT_CLASS = "OBJECT_CLASS"
    OBJECT_MASK = "OBJECT_MASK"
