// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: LGPL-2.1

const PUBLIC_IP = ""; // ADD your ec2-instance public IP
export const PROCESS_UTTERANCE_URL = "http://" + PUBLIC_IP + ":11000/process_utterance";
export const GET_CDFS_URL = "http://" + PUBLIC_IP + ":11000/get_cdfs";
export const START_GAME_URL = "http://" + PUBLIC_IP + ":11000/start_game";
export const BEGIN_SESSION_URL = "http://" + PUBLIC_IP + ":11000/begin_session";
export const STREAMING_URL = "http://" + PUBLIC_IP + ":81";
