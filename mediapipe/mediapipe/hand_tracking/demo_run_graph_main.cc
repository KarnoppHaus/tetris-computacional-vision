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
// An example of sending OpenCV webcam frames into a MediaPipe graph.

#include <cstdlib>
//libs para integração
#include <fstream>
#include <iostream>
#include <cmath>
//
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

//Landmark protobuf
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kLandmarksStream[] = "landmarks";

//função para enviar comando via FIFO
//basicamente cria um fifo que fica publicando o comando enquanto o tetris le esse comando
void sendCommandToTetris(const std::string& command) {
    static std::ofstream fifo("/tmp/tetris_pipe", std::ios::out | std::ios::app);
    if (fifo.is_open()) {
        fifo << command << std::endl;
        fifo.flush();
    } else {
        std::cerr << "Erro ao abrir o pipe para escrever.\n";
    }
}

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ABSL_LOG(INFO) << "Initialize the camera or load the video.";
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
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmark_poller,
                      graph.AddOutputStreamPoller(kLandmarksStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        ABSL_LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      ABSL_LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, 1);  // Flip horizontal for selfie view
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
    if (landmark_poller.QueueSize() > 0) {
      if (landmark_poller.Next(&landmark_packet)) {
        auto& landmarks = landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
        //logica para detecção a partir daqui
        //deixar melhor e mais precisa
        for (int hand_i = 0; hand_i < landmarks.size(); ++hand_i) {
          const auto& hand_landmarks = landmarks[hand_i];

          const auto& palma = hand_landmarks.landmark(0);
          const auto& indicador = hand_landmarks.landmark(8);
          const auto& polegar = hand_landmarks.landmark(4);
          const auto& medio = hand_landmarks.landmark(12);
          const auto& anelar = hand_landmarks.landmark(16);
          const auto& mindinho = hand_landmarks.landmark(20);

          auto distancia = [](const auto& a, const auto& b) {
            float dx = a.x() - b.x();
            float dy = a.y() - b.y();
            float dz = a.z() - b.z();
            return std::sqrt(dx * dx + dy * dy + dz * dz);
          };

          float palmaIndicador = distancia(palma, indicador);
          float palmaPolegar = distancia(palma, polegar);
          float palmaMedio = distancia(palma, medio);
          float palmaAnelar = distancia(palma, anelar);
          float palmaMindinho = distancia(palma, mindinho);

          int dedosEstendidos = 0;
          if (palmaIndicador > 0.2f) dedosEstendidos++;
          if (palmaMedio > 0.2f) dedosEstendidos++;
          if (palmaAnelar > 0.2f) dedosEstendidos++;
          if (palmaMindinho > 0.2f) dedosEstendidos++;
          if (palmaPolegar > 0.2f) dedosEstendidos++;

          // Gesto: mão aberta -> DROP
          if (dedosEstendidos >= 5) {
            sendCommandToTetris("DROP");//comando q publica no FIFO
            continue;
          }

          // Gesto: mão fechada -> ROTATE
          if (dedosEstendidos == 0) {
            sendCommandToTetris("ROTATE");
            continue;
          }

          // Gesto: apenas indicador estendido para esquerda/direita
          if (palmaIndicador > 0.2f &&
              palmaMedio < 0.1f &&
              palmaAnelar < 0.1f &&
              palmaMindinho < 0.1f) {
            if (indicador.x() < palma.x() - 0.05f) {
              sendCommandToTetris("LEFT");
            } else if (indicador.x() > palma.x() + 0.05f) {
              sendCommandToTetris("RIGHT");
            }
          }
        }
      }
    }

    auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    if (save_video) {
      if (!writer.isOpened()) {
        ABSL_LOG(INFO) << "Prepare video writer.";
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

  ABSL_LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  }
  ABSL_LOG(INFO) << "Success!";
  return EXIT_SUCCESS;
}