# Caminhos
TETRIS_SRC = game.cpp block.cpp blocks.cpp colors.cpp grid.cpp main.cpp position.cpp
TETRIS_EXE = game
PIPE_PATH = /tmp/tetris_pipe
PID_FILE = tetris.pid
MEDIAPIPE_DIR = mediapipe
MEDIAPIPE_BUILD_TARGET = mediapipe/hand_tracking/hand_tracking:hand_tracking_cpu
MEDIAPIPE_GRAPH_CONFIG = mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

.PHONY: all run tetris mediapipe tmp run_mediapipe run_all clean kill_tetris

# Compila o Tetris
tetris:
	g++ $(TETRIS_SRC) -o $(TETRIS_EXE) -lraylib

# Cria o FIFO (só se não existir)
tmp:
	test -p $(PIPE_PATH) || mkfifo $(PIPE_PATH)

# Compila o MediaPipe
mediapipe:
	cd $(MEDIAPIPE_DIR) && \
	bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 $(MEDIAPIPE_BUILD_TARGET)

# Executa o Tetris
run:
	./$(TETRIS_EXE)

# Executa o MediaPipe
run_mediapipe:
	cd $(MEDIAPIPE_DIR) && \
	GLOG_logtostderr=1 \
	bazel-bin/mediapipe/hand_tracking/hand_tracking/hand_tracking_cpu \
	--calculator_graph_config_file=$(MEDIAPIPE_GRAPH_CONFIG)

# Prepara tudo (compila o tetris, cria o pipe)
all: tmp tetris

# Executa Tetris e MediaPipe (em paralelo, Tetris em background com PID salvo)
run_all: all
	@echo "Iniciando Tetris e MediaPipe..."
	@echo "Pressione Ctrl+C para encerrar"
	@./$(TETRIS_EXE) & echo $$! > $(PID_FILE)
	@cd $(MEDIAPIPE_DIR) && \
	GLOG_logtostderr=1 \
	bazel-bin/mediapipe/hand_tracking/hand_tracking/hand_tracking_cpu \
	--calculator_graph_config_file=$(MEDIAPIPE_GRAPH_CONFIG)

# Mata o processo do Tetris se ainda estiver rodando
kill_tetris:
	-@kill `cat $(PID_FILE)` 2>/dev/null || true
	@rm -f $(PID_FILE)

# Limpa binários e pipe
clean:
	rm -f $(TETRIS_EXE)
	rm -f $(PIPE_PATH)
	rm -f $(PID_FILE)