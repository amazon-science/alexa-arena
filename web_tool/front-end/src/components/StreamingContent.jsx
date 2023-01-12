// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1

import React from 'react';

// import { Container, Header, Grid, Box, SpaceBetween } from '@awsui/components-react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Grid from '@cloudscape-design/components/grid';
import Box from '@cloudscape-design/components/box';
import SpaceBetween from '@cloudscape-design/components/space-between';

import {DropDown} from './DropDown'
import {PROCESS_UTTERANCE_URL, STREAMING_URL, BEGIN_SESSION_URL} from './Urls'

import '../styles/streamingcontentpage.scss';

class Groupchat extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      messageText: null,
      groupMessage: [],
      uid: 'user'
    };
  }
  sendMessage = () => {
    let message = { uid: this.state.uid, text: this.state.messageText };
    this.setState(
      prevState => ({
        groupMessage: [...prevState.groupMessage, message],
        messageText: null
      }),
      () => {
        this.scrollToBottom();
      }
    );
    console.log('Sending message: ' + message.text);
    let data = { utterance: message.text};
    this.sendPostRequest(PROCESS_UTTERANCE_URL, data);
  };
  scrollToBottom = () => {
    const chatList = document.getElementById('chatList');
    chatList.scrollTop = chatList.scrollHeight;
  };

  handleChange = event => {
    this.setState({ messageText: event.target.value });
  };

  handleSubmit = event => {
    event.preventDefault();
    this.sendMessage();
    // @ts-ignore
    event.target.reset();
  };

  sendPostRequest = (url, data) => {
    fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': 'true'
      },
      body: JSON.stringify(data)
    })
      .then(response => {
        if (!response.ok) console.log(response);
        else return response.json();
      })
      .then(data => {
        console.log(data);
        this.setState(
          prevState => ({
            groupMessage: [...prevState.groupMessage, data],
            messageText: null
          }),
          () => {
            this.scrollToBottom();
          }
        );
      });
  };

  componentDidMount() {
    let data = { utterance: 'Rotate Left' };
    this.sendPostRequest(BEGIN_SESSION_URL, data);
  }

  render() {
    return (
      <div>
        <div className="chatWindow">
          <ul className="chat" id="chatList">
            {this.state.groupMessage.map((data, index) => (
              <div>
                {this.state.uid === data.uid ? (
                  <li key={index} className="self">
                    <div className="msg">
                      <p>You</p>
                      <div className="message"> {data.text}</div>
                    </div>
                  </li>
                ) : (
                  <li key={index} className="other">
                    <div className="msg">
                      <p>Robot</p>
                      <div className="message"> {data.text} </div>
                    </div>
                  </li>
                )}
              </div>
            ))}
          </ul>
        </div>
        <div className="chatInputWrapper">
          <form onSubmit={this.handleSubmit}>
            <input
              className="textarea input"
              type="text"
              placeholder="Enter your instruction..."
              onChange={this.handleChange}
            />
          </form>
        </div>
      </div>
    );
  }
}

export function StreamingContent() {
  return (
    <div>
      <Box padding={{ top: 'xxl', horizontal: 's', bottom: 'l' }}>
        <Grid
          gridDefinition={[
            { colspan: { xl: 9, l: 9, s: 9, xxs: 9 }, offset: { l: 0, xxs: 0 } },
            { colspan: { xl: 3, l: 3, s: 3, xxs: 3 }, offset: { s: 0, xxs: 0 } }
          ]}
        >
          <SpaceBetween size="l">
            <div>
              <Container header={
                <Grid
                    gridDefinition={[
                      { colspan: { xl: 6, l: 6, s: 6, xxs: 6 }, offset: { l: 0, xxs: 0 } },
                      { colspan: { xl: 6, l: 6, s: 6, xxs: 6 }, offset: { s: 0, xxs: 0 } }
                    ]}
                >
                  <Header variant="h2">Arena Simulator</Header>
                  <DropDown />
                </Grid>
              }>
                <iframe
                  src={STREAMING_URL}
                  sandbox="allow-scripts allow-same-origin"
                  width="100%"
                  height="820px"
                />
              </Container>
            </div>
          </SpaceBetween>
          <SpaceBetween size="l">
            <Container header={<Header variant="h2">Chat with Robot</Header>}>
              <Groupchat />
            </Container>
          </SpaceBetween>
        </Grid>
      </Box>
    </div>
  );
}
