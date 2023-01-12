// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1


import React from 'react';
import { Route } from 'react-router-dom';

import PlayConsole from "./PlayConsole.jsx";

// Class App is the "output" generated on every build,
// it is what you will see on the webpage.
export default class App extends React.Component {
  render() {
    return (
      // When you create a new file or template, add it below
      // as a new 'Route' so you can link to it with a url.

      <div>
        <Route exact path="/" component={PlayConsole} />
      </div>
    );
  }
}
