#include <windows.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <cstring>
#include <cstdint>

#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

namespace mp_hl = mediapipe::tasks::vision::hand_landmarker;

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }

static std::atomic<bool>  g_running{false};
static std::thread        g_thread;

static double   g_lm[63] = {};
static int      g_lm_count = 0;
static std::mutex g_lm_mx;

static uint8_t  g_frame[320 * 240 * 3] = {};
static int      g_frame_w = 0, g_frame_h = 0;
static std::mutex g_frame_mx;

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void tracker_loop() {
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);

    auto opts = std::make_unique<mp_hl::HandLandmarkerOptions>();
    opts->base_options.model_asset_path        = "hand_landmarker.task";
    opts->running_mode                         = mediapipe::tasks::vision::core::RunningMode::VIDEO;
    opts->num_hands                            = 1;
    opts->min_hand_detection_confidence        = 0.5f;
    opts->min_hand_presence_confidence         = 0.5f;
    opts->min_tracking_confidence              = 0.5f;

    auto lm_or = mp_hl::HandLandmarker::Create(std::move(opts));
    if (!lm_or.ok()) return;
    auto landmarker = std::move(lm_or.value());

    int64_t last_ts = -1;
    while (g_running.load()) {
        cv::Mat bgr;
        if (!cap.read(bgr) || bgr.empty()) continue;
        cv::flip(bgr, bgr, 1);

        {
            std::lock_guard<std::mutex> lk(g_frame_mx);
            g_frame_w = bgr.cols;
            g_frame_h = bgr.rows;
            if (bgr.isContinuous()) {
                std::memcpy(g_frame, bgr.data, bgr.total() * 3);
            } else {
                cv::Mat cont;
                bgr.copyTo(cont);
                std::memcpy(g_frame, cont.data, cont.total() * 3);
            }
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        auto mp_frame = std::make_shared<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB,
            rgb.cols, rgb.rows,
            static_cast<uint32_t>(rgb.step),
            rgb.data,
            [](uint8_t*) {}
        );
        mediapipe::Image mp_image(mp_frame);

        int64_t ts = now_ms();
        if (ts <= last_ts) ts = last_ts + 1;
        last_ts = ts;

        auto res_or = landmarker->DetectForVideo(mp_image, ts);

        std::lock_guard<std::mutex> lk(g_lm_mx);
        if (res_or.ok() && !res_or.value().hand_landmarks.empty()) {
            auto& lms = res_or.value().hand_landmarks[0].landmarks;
            g_lm_count = 63;
            for (int i = 0; i < 21; ++i) {
                g_lm[i * 3]     = static_cast<double>(lms[i].x);
                g_lm[i * 3 + 1] = static_cast<double>(lms[i].y);
                g_lm[i * 3 + 2] = static_cast<double>(lms[i].z);
            }
        } else {
            g_lm_count = 0;
        }
    }

    landmarker->Close().IgnoreError();
    cap.release();
}

extern "C" {

__declspec(dllexport) int ht_start() {
    if (g_running.load()) return 0;
    g_running.store(true);
    g_thread = std::thread(tracker_loop);
    return 1;
}

__declspec(dllexport) void ht_stop() {
    g_running.store(false);
    if (g_thread.joinable()) g_thread.join();
}

__declspec(dllexport) int ht_get_landmarks(double* out, int buf_size) {
    std::lock_guard<std::mutex> lk(g_lm_mx);
    if (g_lm_count == 0 || buf_size < 63) return 0;
    std::memcpy(out, g_lm, 63 * sizeof(double));
    return 63;
}

__declspec(dllexport) int ht_get_frame(uint8_t* out, int* w, int* h) {
    std::lock_guard<std::mutex> lk(g_frame_mx);
    if (g_frame_w == 0) return 0;
    std::memcpy(out, g_frame, g_frame_w * g_frame_h * 3);
    *w = g_frame_w;
    *h = g_frame_h;
    return 1;
}

}
