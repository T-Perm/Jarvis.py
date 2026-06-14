// hand_tracker.cc
// Python-callable DLL: runs MediaPipe HandLandmarker on a background thread,
// exposes landmarks and raw BGR frame via simple C exports.
//
// Build (Bazel, add to your existing MediaPipe workspace BUILD):
//   cc_binary(
//       name = "hand_tracker",
//       srcs = ["hand_tracker.cc"],
//       linkshared = True,
//       linkstatic = True,
//       deps = [
//           "@mediapipe//mediapipe/tasks/cc/vision/hand_landmarker",
//           "@mediapipe//mediapipe/framework/formats:image_frame",
//           "@mediapipe//mediapipe/framework/formats:image",
//           "@mediapipe//mediapipe/framework/port:opencv_video_inc",
//           "@mediapipe//mediapipe/framework/port:opencv_imgproc_inc",
//       ],
//   )
//
// Then copy hand_tracker.dll next to app.py.

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

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
static std::atomic<bool>  g_running{false};
static std::thread        g_thread;

// Landmarks: 21 * 3 doubles (x, y, z).  g_lm_count == 0 means no hand.
static double   g_lm[63] = {};
static int      g_lm_count = 0;
static std::mutex g_lm_mx;

// Frame: raw BGR bytes at capture resolution.
static uint8_t  g_frame[320 * 240 * 3] = {};
static int      g_frame_w = 0, g_frame_h = 0;
static std::mutex g_frame_mx;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Tracker thread
// ---------------------------------------------------------------------------
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

        // Share frame
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

        // Inference
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

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
extern "C" {

// Start the tracker thread. Returns 1 on success, 0 if already running.
__declspec(dllexport) int ht_start() {
    if (g_running.load()) return 0;
    g_running.store(true);
    g_thread = std::thread(tracker_loop);
    return 1;
}

// Stop the tracker thread and join.
__declspec(dllexport) void ht_stop() {
    g_running.store(false);
    if (g_thread.joinable()) g_thread.join();
}

// Copy the latest landmarks into `out` (must be double[63]).
// Returns 63 if a hand was detected, 0 otherwise.
__declspec(dllexport) int ht_get_landmarks(double* out, int buf_size) {
    std::lock_guard<std::mutex> lk(g_lm_mx);
    if (g_lm_count == 0 || buf_size < 63) return 0;
    std::memcpy(out, g_lm, 63 * sizeof(double));
    return 63;
}

// Copy the latest BGR frame into `out` (must be uint8[320*240*3]).
// Fills *w and *h with actual frame size. Returns 1 on success, 0 if no frame yet.
__declspec(dllexport) int ht_get_frame(uint8_t* out, int* w, int* h) {
    std::lock_guard<std::mutex> lk(g_frame_mx);
    if (g_frame_w == 0) return 0;
    std::memcpy(out, g_frame, g_frame_w * g_frame_h * 3);
    *w = g_frame_w;
    *h = g_frame_h;
    return 1;
}

} // extern "C"
