// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1

import React from 'react';
import {AppLayout} from '@cloudscape-design/components';

import {StreamingContent} from "./StreamingContent";

export default function PlayConsole() {
  return (
    <AppLayout
      content={<StreamingContent/>}
      contentType="default"
      disableContentPaddings={true}
      navigationHide={true}
      toolsHide={true}
    />
  );
}
