// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1

import React from "react";

import Grid from '@cloudscape-design/components/grid';
import Button from '@cloudscape-design/components/button';
import ButtonDropdown from '@cloudscape-design/components/button-dropdown';
import {GET_CDFS_URL, START_GAME_URL} from './Urls'

export class DropDown extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            challengeDefinitions: [{id:"123", text:"Mission1", cdf:""},{id:"13", text:"Mission2", cdf:""}],
            currentMission: null
        };
    }

    fetchChallengeDefinitions = () => {
        fetch(GET_CDFS_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': 'true'
            }
        })
        .then(response => {
            if (!response.ok) console.log(response);
            else return response.json();
        })
        .then(data => {
            this.setState({challengeDefinitions: data["cdfs"], currentMission: data["cdfs"][0].id});
        });
    };

    startGame = () => {
        let cdf_json = this.state.challengeDefinitions.filter(cdf => {
            return cdf.id === this.state.currentMission
        });
        let data = {cdf_data: cdf_json[0].cdf};
        fetch(START_GAME_URL, {
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
        });
    };

    componentDidMount() {
        this.fetchChallengeDefinitions();
        this.setState({currentMission: this.state.challengeDefinitions[0].id});
    }

    handleItemClick = event => {
        this.setState({currentMission: event.detail.id});
    };

    handleButtonClick = event => {
        console.log("Button click");
        this.startGame();
    }

    render() {
        return (
            <Grid
                gridDefinition={[
                    { colspan: { xl: 8, l: 8, s: 8, xxs: 8 }, offset: { l: 0, xxs: 0 } },
                    { colspan: { xl: 4, l: 4, s: 4, xxs: 4 }, offset: { s: 0, xxs: 0 } }
                ]}
            >
                <ButtonDropdown
                    id="button-dropdown-mission"
                    items={this.state.challengeDefinitions}
                    onItemClick={this.handleItemClick}
                >
                    Game mission: {this.state.currentMission}
                </ButtonDropdown>
                <Button variant="primary" onClick={this.handleButtonClick}>Launch game</Button>
            </Grid>
        )
    }
}