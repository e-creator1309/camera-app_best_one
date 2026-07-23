/*
 * doc_scan.cpp — document quadrilateral detector using OpenCV.
 *
 * Replaces the hand-rolled BFS pipeline with:
 *   1. Subsample camera Y-plane → SCAN_W×SCAN_H cv::Mat
 *   2. GaussianBlur → Canny edge detection
 *   3. Dilate to close small border gaps
 *   4. findContours → approxPolyDP to enumerate candidate quads
 *   5. Validate each 4-vertex polygon (convexity, area, aspect ratio,
 *      side lengths, interior angles) — same thresholds as before
 *   6. Return the largest valid quad, corners normalised to [0..1]
 *
 * JNI interface is identical to the previous doc_scan.c — no Kotlin changes
 * are needed.
 *
 * THREAD SAFETY
 * All state is stack/heap-local per call (cv::Mat is stack-allocated).
 * Safe to call from a single-thread Executor as before.
 */

#include <jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <algorithm>
#include <vector>

#define LOG_TAG "DocScan"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/* ── Working resolution ─────────────────────────────────────────────────── */
static constexpr int SCAN_W = 160;
static constexpr int SCAN_H = 214;

/* ── Tuning constants ───────────────────────────────────────────────────── */
static constexpr float MIN_AREA_FRAC  = 0.09f;
static constexpr float MAX_AREA_FRAC  = 0.94f;
static constexpr float MIN_ANGLE_DEG  = 45.0f;
static constexpr float MAX_ANGLE_DEG  = 135.0f;
static constexpr float MIN_SIDE_FRAC  = 0.07f;
static constexpr float MIN_BBOX_RATIO = 0.25f;

/* ── Geometry helpers ────────────────────────────────────────────────────── */

/* Interior angle at vertex B (degrees) in the path A→B→C. */
static float interior_angle(cv::Point a, cv::Point b, cv::Point c) {
    float ux = (float)(a.x - b.x), uy = (float)(a.y - b.y);
    float vx = (float)(c.x - b.x), vy = (float)(c.y - b.y);
    float lu = std::sqrt(ux*ux + uy*uy);
    float lv = std::sqrt(vx*vx + vy*vy);
    if (lu < 0.5f || lv < 0.5f) return 0.f;
    float cosA = std::max(-1.f, std::min(1.f, (ux*vx + uy*vy) / (lu * lv)));
    return std::acos(cosA) * (180.f / (float)CV_PI);
}

/* Return true if q is a valid document quad inside a W×H image. */
static bool is_valid_quad(const std::vector<cv::Point>& q, int W, int H) {
    if (q.size() != 4) return false;

    /* Convexity */
    if (!cv::isContourConvex(q)) return false;

    /* Area fraction */
    double area = cv::contourArea(q);
    double frac = area / (double)(W * H);
    if (frac < MIN_AREA_FRAC || frac > MAX_AREA_FRAC) return false;

    /* Bounding-box aspect ratio — rejects strips (spiral binding etc.) */
    cv::Rect bb = cv::boundingRect(q);
    float bbRatio = (float)std::min(bb.width, bb.height)
                  / (float)std::max(bb.width, bb.height);
    if (bbRatio < MIN_BBOX_RATIO) return false;

    /* Minimum side length */
    float diagLen = std::sqrt((float)(W*W + H*H));
    float minSide = MIN_SIDE_FRAC * diagLen;
    for (int i = 0; i < 4; i++) {
        float dx = (float)(q[i].x - q[(i+1)%4].x);
        float dy = (float)(q[i].y - q[(i+1)%4].y);
        if (std::sqrt(dx*dx + dy*dy) < minSide) return false;
    }

    /* Interior angles 45°–135° */
    for (int i = 0; i < 4; i++) {
        float ang = interior_angle(q[(i+3)%4], q[i], q[(i+1)%4]);
        if (ang < MIN_ANGLE_DEG || ang > MAX_ANGLE_DEG) return false;
    }

    return true;
}

/* Reorder four corners into [TL, TR, BR, BL] using sum/difference heuristic. */
static std::vector<cv::Point> order_corners(std::vector<cv::Point> pts) {
    /* Sort by (x+y): smallest = TL, largest = BR */
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point& a, const cv::Point& b){
                  return (a.x + a.y) < (b.x + b.y);
              });
    std::vector<cv::Point> out(4);
    out[0] = pts[0]; /* TL */
    out[2] = pts[3]; /* BR */
    /* Of the two middle points, larger (x−y) → TR, smaller → BL */
    if ((pts[1].x - pts[1].y) < (pts[2].x - pts[2].y)) {
        out[1] = pts[2]; out[3] = pts[1];
    } else {
        out[1] = pts[1]; out[3] = pts[2];
    }
    return out;
}

/* ── JNI exports ─────────────────────────────────────────────────────────── */

extern "C" {

/*
 * docScanFindQuadNative — find the best document quad in a camera Y-plane.
 * corners[8] = [tlX,tlY, trX,trY, brX,brY, blX,blY] in [0..1] normalised.
 */
JNIEXPORT jboolean JNICALL
Java_com_replit_cameraapp_NativeImaging_docScanFindQuadNative(
        JNIEnv *env, jobject /*thiz*/,
        jbyteArray yPlane_jni, jint rowStride,
        jint cropL, jint cropT, jint cropW, jint cropH,
        jfloatArray corners_jni)
{
    if (env->GetArrayLength(corners_jni) < 8) return JNI_FALSE;
    if (cropW <= 0 || cropH <= 0)             return JNI_FALSE;

    jbyte *yRaw = env->GetByteArrayElements(yPlane_jni, nullptr);
    if (!yRaw) return JNI_FALSE;

    /* Subsample the Y-plane into a SCAN_W×SCAN_H grayscale Mat. */
    cv::Mat scan(SCAN_H, SCAN_W, CV_8UC1);
    for (int sy = 0; sy < SCAN_H; sy++) {
        int fy = cropT + (int)((long)sy * (long)cropH / SCAN_H);
        auto *dst = scan.ptr<uint8_t>(sy);
        const auto *src = reinterpret_cast<const uint8_t*>(yRaw)
                          + (long)fy * rowStride;
        for (int sx = 0; sx < SCAN_W; sx++) {
            int fx = cropL + (int)((long)sx * (long)cropW / SCAN_W);
            dst[sx] = src[fx];
        }
    }
    env->ReleaseByteArrayElements(yPlane_jni, yRaw, JNI_ABORT);

    /* Blur → Canny edges → dilate to close small gaps. */
    cv::Mat blurred, edges;
    cv::GaussianBlur(scan, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 30, 90);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 1);

    /* Find contours and approximate each to a polygon. */
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const double minArea = MIN_AREA_FRAC * (double)(SCAN_W * SCAN_H);
    float bestScore = -1.f;
    std::vector<cv::Point> bestQuad;

    for (auto& contour : contours) {
        if (cv::contourArea(contour) < minArea) continue;

        double peri = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * peri, true);

        if (approx.size() != 4) continue;
        if (!is_valid_quad(approx, SCAN_W, SCAN_H)) continue;

        float score = (float)cv::contourArea(approx);
        if (score > bestScore) { bestScore = score; bestQuad = approx; }
    }

    if (bestQuad.empty()) return JNI_FALSE;

    std::vector<cv::Point> ordered = order_corners(bestQuad);
    float corners[8] = {
        (float)ordered[0].x / SCAN_W, (float)ordered[0].y / SCAN_H,
        (float)ordered[1].x / SCAN_W, (float)ordered[1].y / SCAN_H,
        (float)ordered[2].x / SCAN_W, (float)ordered[2].y / SCAN_H,
        (float)ordered[3].x / SCAN_W, (float)ordered[3].y / SCAN_H,
    };
    env->SetFloatArrayRegion(corners_jni, 0, 8, corners);
    return JNI_TRUE;
}

/*
 * docScanSmoothCornersNative — single EMA step for corner smoothing.
 * out[i] = prev[i] + alpha * (curr[i] − prev[i]).  Call at ~8–12 fps.
 */
JNIEXPORT void JNICALL
Java_com_replit_cameraapp_NativeImaging_docScanSmoothCornersNative(
        JNIEnv *env, jobject /*thiz*/,
        jfloatArray prev_jni, jfloatArray curr_jni,
        jfloat alpha,
        jfloatArray out_jni)
{
    if (env->GetArrayLength(prev_jni) < 8 ||
        env->GetArrayLength(curr_jni) < 8 ||
        env->GetArrayLength(out_jni)  < 8) return;

    float prev[8], curr[8], out[8];
    env->GetFloatArrayRegion(prev_jni, 0, 8, prev);
    env->GetFloatArrayRegion(curr_jni, 0, 8, curr);
    for (int i = 0; i < 8; i++)
        out[i] = prev[i] + (float)alpha * (curr[i] - prev[i]);
    env->SetFloatArrayRegion(out_jni, 0, 8, out);
}

/*
 * docScanIsValidQuadNative — re-validates a normalised quad after EMA
 * smoothing using the same geometry checks as the detector.
 */
JNIEXPORT jboolean JNICALL
Java_com_replit_cameraapp_NativeImaging_docScanIsValidQuadNative(
        JNIEnv *env, jobject /*thiz*/,
        jfloatArray corners_jni, jint frameW, jint frameH)
{
    if (env->GetArrayLength(corners_jni) < 8) return JNI_FALSE;
    float nc[8];
    env->GetFloatArrayRegion(corners_jni, 0, 8, nc);

    std::vector<cv::Point> pts(4);
    for (int i = 0; i < 4; i++) {
        pts[i] = cv::Point(
            (int)(nc[i*2]   * (float)frameW + 0.5f),
            (int)(nc[i*2+1] * (float)frameH + 0.5f)
        );
    }
    return is_valid_quad(pts, (int)frameW, (int)frameH) ? JNI_TRUE : JNI_FALSE;
}

} /* extern "C" */
