#!/bin/bash
cd mediapipe
env -i DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY HOME=$HOME PATH=$PATH \
bazel-bin/mediapipe/hand_tracking/hand_tracking/hand_tracking_cpu \
--calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

XXxxxxx
