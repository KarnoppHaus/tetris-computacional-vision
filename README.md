# tetris-computacional-vision
Jogo clássico Tetris utilizando visão computacional para executar comandos do jogo.

Código para rodar no mediapipe:

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/hand_tracking/hand_tracking:hand_tracking_cpu

GLOG_logtostderr=1 bazel-bin/mediapipe/hand_tracking/hand_tracking/hand_tracking_cpu   --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

