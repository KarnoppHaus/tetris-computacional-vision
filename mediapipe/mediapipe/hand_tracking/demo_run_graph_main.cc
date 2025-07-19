// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Exemplo para enviar frames da webcam OpenCV e comandos para o Tetris via FIFO.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <map>

#include <fcntl.h>     // open flags
#include <sys/stat.h>  // mkfifo
#include <unistd.h>    // open, close, write
#include <sys/mman.h>  // mmap, shm_open
#include <sys/stat.h>  // For mode constants
#include <fcntl.h>     // For O_* constants
#include <cstring>     // memcpy

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kLandmarksStream[] = "landmarks";

static constexpr char kTetrisPipe[] = "/tmp/tetris_pipe";
static constexpr char kFramePipe[] = "/tmp/frame_pipe";

static constexpr char kSharedMemoryName[] = "/frame_shm";
static constexpr size_t kSharedMemorySize = 1024 * 1024;  // 1MB

// Envia comando para o Tetris via FIFO (pipe)
void sendCommandToTetris(const std::string& command) {
    static std::ofstream fifo(kTetrisPipe, std::ios::out | std::ios::app);
    if (fifo.is_open()) {
        fifo << command << std::endl;
        fifo.flush();
    } else {
        std::cerr << "Erro ao abrir pipe " << kTetrisPipe << " para escrever comandos.\n";
    }
}

// Controla cooldown para não enviar comando repetido muito rápido
class CommandCooldown {
public:
    CommandCooldown(int cooldown_ms = 500) : cooldown_ms_(cooldown_ms) {}

    bool canSend(const std::string& cmd) {
        auto now = std::chrono::steady_clock::now();
        if (lastCommandTime_.find(cmd) == lastCommandTime_.end() ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - lastCommandTime_[cmd]).count() > cooldown_ms_) {
            lastCommandTime_[cmd] = now;
            return true;
        }
        return false;
    }
private:
    int cooldown_ms_;
    std::map<std::string, std::chrono::steady_clock::time_point> lastCommandTime_;
};

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Arquivo contendo a configuração do grafo MediaPipe.");
ABSL_FLAG(std::string, input_video_path, "",
          "Caminho completo do vídeo. Se vazio, usa webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Caminho para salvar o vídeo de saída (.mp4). Se vazio, mostra janela.");

absl::Status RunMPPGraph() {
    int shm_fd;
    void* shm_ptr = nullptr;
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
        absl::GetFlag(FLAGS_calculator_graph_config_file),
        &calculator_graph_config_contents));
    ABSL_LOG(INFO) << "Carregou config do grafo MediaPipe.";
    mediapipe::CalculatorGraphConfig config =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    // Abre câmera ou vídeo
    cv::VideoCapture capture;
    const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
    if (load_video) {
        capture.open(absl::GetFlag(FLAGS_input_video_path));
    } else {
        capture.open(0);
    }
    RET_CHECK(capture.isOpened());

    cv::VideoWriter writer;
    const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();

    if (!save_video) {
        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        capture.set(cv::CAP_PROP_FPS, 30);
#endif
    }

    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                        graph.AddOutputStreamPoller(kOutputStream));
    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmark_poller,
                        graph.AddOutputStreamPoller(kLandmarksStream));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    CommandCooldown cooldown(500);
    bool grab_frames = true;
    int currentPieceColumn = 4;
    const int totalColumns = 10;

    while (grab_frames) {
        int shm_fd = shm_open(kSharedMemoryName, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            std::cerr << "❌ Falha ao criar memória compartilhada.\n";
            return absl::UnknownError("shm_open failed");
        }
        ftruncate(shm_fd, kSharedMemorySize);

        void* shm_ptr = mmap(0, kSharedMemorySize, PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shm_ptr == MAP_FAILED) {
            std::cerr << "❌ Falha ao mapear memória compartilhada.\n";
            return absl::UnknownError("mmap failed");
        }

        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty()) {
            if (!load_video) {
                ABSL_LOG(INFO) << "Ignorando frame vazio da câmera.";
                continue;
            }
            ABSL_LOG(INFO) << "Fim do vídeo.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
        if (!load_video) {
            cv::flip(camera_frame, camera_frame, 1);  // espelho selfie
        }

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us =
            (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
            kInputStream, mediapipe::Adopt(input_frame.release())
                              .At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet packet;
        if (!poller.Next(&packet)) break;

        mediapipe::Packet landmark_packet;
        if (landmark_poller.QueueSize() > 0 && landmark_poller.Next(&landmark_packet)) {
            auto& landmarks = landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

            for (const auto& hand_landmarks : landmarks) {
                const mediapipe::NormalizedLandmark& palma     = hand_landmarks.landmark(0);
                const mediapipe::NormalizedLandmark& indicador = hand_landmarks.landmark(8);
                const mediapipe::NormalizedLandmark& polegar   = hand_landmarks.landmark(4);

                auto distancia = [](const mediapipe::NormalizedLandmark& a, const mediapipe::NormalizedLandmark& b) {
                    float dx = a.x() - b.x();
                    float dy = a.y() - b.y();
                    float dz = a.z() - b.z();
                    return std::sqrt(dx * dx + dy * dy + dz * dz);
                };

                // --- ROTATE ---
                bool indicador_estendido = distancia(palma, indicador) > 0.20f;
                bool polegar_estendido   = distancia(palma, polegar) > 0.20f;
                bool polegar_lateral     = polegar.x() < palma.x() - 0.03f;

                bool medio_flexionado    = distancia(palma, hand_landmarks.landmark(12)) < 0.15f;
                bool anelar_flexionado   = distancia(palma, hand_landmarks.landmark(16)) < 0.15f;
                bool minimo_flexionado   = distancia(palma, hand_landmarks.landmark(20)) < 0.15f;

                if (indicador_estendido && polegar_estendido && polegar_lateral &&
                    medio_flexionado && anelar_flexionado && minimo_flexionado &&
                    cooldown.canSend("ROTATE")) {
                    sendCommandToTetris("ROTATE");
                    std::cout << "🔄 Gesto detectado: ROTATE\n";
                    continue;
                }

                // --- DROP ---
                int dedosEstendidos = 0;
                auto palma_to_dedo = [&](int idx) {
                    return distancia(palma, hand_landmarks.landmark(idx)) > 0.2f;
                };
                if (palma_to_dedo(8)) dedosEstendidos++;
                if (palma_to_dedo(12)) dedosEstendidos++;
                if (palma_to_dedo(16)) dedosEstendidos++;
                if (palma_to_dedo(20)) dedosEstendidos++;
                if (palma_to_dedo(4))  dedosEstendidos++;

                if (dedosEstendidos >= 5 && cooldown.canSend("DROP")) {
                    sendCommandToTetris("DROP");
                    std::cout << "🟢 Gesto detectado: DROP\n";
                    continue;
                }

                // --- LEFT / RIGHT ---
                float indicadorX = indicador.x();
                int targetColumn = static_cast<int>(indicadorX * totalColumns);
                targetColumn = std::clamp(targetColumn, 0, totalColumns - 1);

                if (targetColumn > currentPieceColumn) {
                    if (cooldown.canSend("RIGHT")) {
                        sendCommandToTetris("RIGHT");
                        currentPieceColumn++;
                        std::cout << "➡️ Enviando RIGHT para coluna " << currentPieceColumn << "\n";
                    }
                } else if (targetColumn < currentPieceColumn) {
                    if (cooldown.canSend("LEFT")) {
                        sendCommandToTetris("LEFT");
                        currentPieceColumn--;
                        std::cout << "⬅️ Enviando LEFT para coluna " << currentPieceColumn << "\n";
                    }
                }
            }
        }

        // Pega frame processado para enviar ao pipe e mostrar
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        // Codifica como JPEG e envia para pipe (se aberto)
        std::vector<uchar> buffer;
        cv::imencode(".jpg", output_frame_mat, buffer);

        int size = static_cast<int>(buffer.size());
        if (size + sizeof(int) < kSharedMemorySize) {
            memcpy(shm_ptr, &size, sizeof(int));                    // primeiro 4 bytes: tamanho
            memcpy(static_cast<char*>(shm_ptr) + sizeof(int),       // após isso: dados
                  buffer.data(), size);
            std::cout << "📤 Frame enviado à memória compartilhada (" << size << " bytes)\n";
        } else {
            std::cerr << "⚠️ Imagem muito grande para memória compartilhada\n";
        }


        if (save_video) {
            if (!writer.isOpened()) {
                ABSL_LOG(INFO) << "Abrindo writer para vídeo de saída.";
                writer.open(absl::GetFlag(FLAGS_output_video_path),
                            mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                            capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
                RET_CHECK(writer.isOpened());
            }
            writer.write(output_frame_mat);
        } else {
            cv::imshow(kWindowName, output_frame_mat);
            const int pressed_key = cv::waitKey(5);
            if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
        }
    }

    ABSL_LOG(INFO) << "Finalizando execução.";
    if (writer.isOpened()) writer.release();
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    munmap(shm_ptr, kSharedMemorySize);
    close(shm_fd);
    shm_unlink(kSharedMemoryName);  // opcional: remove a memória após uso


    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        ABSL_LOG(ERROR) << "Falha ao executar grafo: " << run_status.message();
        return EXIT_FAILURE;
    }
    ABSL_LOG(INFO) << "Execução finalizada com sucesso!";
    return EXIT_SUCCESS;
}
