

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "img_converters.h"

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

// TFLite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "model_data.h"

// =====================================================
// CONFIGURATION
// =====================================================

static const char *TAG = "EMNIST_S3";

// Thay bằng Wi-Fi của bạn
#define WIFI_SSID "Phung"
#define WIFI_PASSWORD "08081975"

#define FRAME_WIDTH 96
#define FRAME_HEIGHT 96
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT)

#define STREAM_QUALITY 75
#define RESULT_THRESHOLD 55.0f

// =====================================================
// CAMERA PIN CONFIGURATION
// Giữ nguyên pin map của code đang chạy
// =====================================================

#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 15
#define SIOD_GPIO_NUM 4
#define SIOC_GPIO_NUM 5

#define Y9_GPIO_NUM 16
#define Y8_GPIO_NUM 17
#define Y7_GPIO_NUM 18
#define Y6_GPIO_NUM 12
#define Y5_GPIO_NUM 10
#define Y4_GPIO_NUM 8
#define Y3_GPIO_NUM 9
#define Y2_GPIO_NUM 11

#define VSYNC_GPIO_NUM 6
#define HREF_GPIO_NUM 7
#define PCLK_GPIO_NUM 13

// =====================================================
// EMNIST LABELS
// =====================================================

const char EMNIST_LABELS[47] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'};

// =====================================================
// TFLITE MICRO
// =====================================================

constexpr int kTensorArenaSize = 90 * 1024;

static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;

TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

// =====================================================
// SHARED DATA
// =====================================================

// Ảnh grayscale mới nhất dành cho HTTP stream
static uint8_t *stream_frame = nullptr;

// Mutex bảo vệ frame và kết quả nhận diện
static SemaphoreHandle_t data_mutex = nullptr;

// Kết quả nhận diện
struct RecognitionResult {
  char character;
  float confidence;
  int64_t latency_ms;
  bool valid;
};

static RecognitionResult latest_result = {'-', 0.0f, 0, false};

// Wi-Fi
static EventGroupHandle_t wifi_event_group = nullptr;

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

static int wifi_retry_count = 0;
#define WIFI_MAX_RETRY 10

// =====================================================
// CAMERA INITIALIZATION
// =====================================================

static esp_err_t init_camera(void) {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;

  // Giữ nguyên grayscale cho EMNIST
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size = FRAMESIZE_96X96;

  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_DRAM;

  return esp_camera_init(&config);
}

// =====================================================
// TFLITE INITIALIZATION
// =====================================================

static bool init_tflm(void) {
  model = tflite::GetModel(emnist_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Schema model version khong khop!");
    return false;
  }

  static tflite::MicroMutableOpResolver<10> resolver;

  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddDequantize();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddQuantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors that bai!");
    return false;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    ESP_LOGE(TAG, "Model khong dung tensor INT8!");
    return false;
  }

  ESP_LOGI(TAG, "TFLite Micro initialized!");

  return true;
}

// =====================================================
// WIFI EVENT HANDLER
// =====================================================

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    ESP_LOGI(TAG, "WiFi started. Connecting...");
    esp_wifi_connect();
  }

  else if (event_base == WIFI_EVENT &&
           event_id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_event_sta_disconnected_t *event =
        (wifi_event_sta_disconnected_t *)event_data;

    ESP_LOGE(TAG, "WiFi disconnected! Reason code: %d", event->reason);

    if (wifi_retry_count < WIFI_MAX_RETRY) {
      wifi_retry_count++;

      ESP_LOGW(TAG, "WiFi reconnect %d/%d", wifi_retry_count, WIFI_MAX_RETRY);

      esp_wifi_connect();
    } else {
      ESP_LOGE(TAG, "WiFi connection failed after maximum retries");

      xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);
    }
  }

  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;

    ESP_LOGI(TAG, "WiFi connected!");
    ESP_LOGI(TAG, "IP: " IPSTR, IP2STR(&event->ip_info.ip));

    wifi_retry_count = 0;

    xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

// =====================================================
// WIFI INITIALIZATION
// =====================================================

static bool init_wifi(void) {
  esp_err_t ret = nvs_flash_init();

  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {

    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }

  ESP_ERROR_CHECK(ret);

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());

  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  wifi_event_group = xEventGroupCreate();

  if (wifi_event_group == nullptr) {
    ESP_LOGE(TAG, "Cannot create WiFi event group");
    return false;
  }

  ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                             &wifi_event_handler, nullptr));

  ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                             &wifi_event_handler, nullptr));

  wifi_config_t wifi_config = {};

  strncpy((char *)wifi_config.sta.ssid, WIFI_SSID,
          sizeof(wifi_config.sta.ssid) - 1);

  strncpy((char *)wifi_config.sta.password, WIFI_PASSWORD,
          sizeof(wifi_config.sta.password) - 1);

  wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));

  ESP_ERROR_CHECK(esp_wifi_start());

  ESP_LOGI(TAG, "Connecting WiFi...");

  EventBits_t bits =
      xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                          pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));

  if (bits & WIFI_CONNECTED_BIT) {
    return true;
  }

  ESP_LOGE(TAG, "WiFi connection failed or timed out");

  return false;
}

// =====================================================
// HTML WEB DASHBOARD
// =====================================================

static const char INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>ESP32-S3 EMNIST Monitor</title>

<style>
body {
    background: #101820;
    color: white;
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 700px;
    margin: auto;
}

h1 {
    color: #00d9a5;
}

.camera {
    width: 100%;
    max-width: 480px;
    border: 3px solid #00d9a5;
    border-radius: 12px;
    background: #000;
}

.result {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 20px;
}

.card {
    background: #1e2b38;
    padding: 20px 5px;
    border-radius: 12px;
}

.value {
    font-size: 28px;
    font-weight: bold;
    color: #00d9a5;
}

.label {
    color: #aab8c2;
    font-size: 13px;
    margin-top: 8px;
}

.status {
    margin: 15px;
    color: #00d9a5;
}

@media(max-width:450px) {
    .value { font-size: 21px; }
    .card { padding: 15px 3px; }
}
</style>
</head>

<body>
<div class="container">

<h1>ESP32-S3 EMNIST</h1>

<div class="status">LIVE CAMERA STREAM</div>

<img class="camera" src="/stream" alt="Camera Stream">

<div class="result">

<div class="card">
<div class="value" id="character">-</div>
<div class="label">PREDICTED CHARACTER</div>
</div>

<div class="card">
<div class="value" id="confidence">0%</div>
<div class="label">CONFIDENCE</div>
</div>

<div class="card">
<div class="value" id="latency">0 ms</div>
<div class="label">INFERENCE LATENCY</div>
</div>

</div>

<div class="status" id="state">Connecting...</div>

</div>

<script>
async function updateStatus() {
    try {
        const response = await fetch('/status', {
            cache: 'no-store'
        });

        if (!response.ok) throw new Error('HTTP error');

        const data = await response.json();

        document.getElementById('character').textContent =
            data.character;

        document.getElementById('confidence').textContent =
            data.confidence.toFixed(1) + '%';

        document.getElementById('latency').textContent =
            data.latency + ' ms';

        document.getElementById('state').textContent =
            data.valid ? 'EMNIST: DETECTED' : 'WAITING FOR RESULT';

    } catch(e) {
        document.getElementById('state').textContent =
            'Connection error';
    }
}

setInterval(updateStatus, 500);
updateStatus();
</script>

</body>
</html>
)rawliteral";

// =====================================================
// HTTP: HOME PAGE
// =====================================================

static esp_err_t index_handler(httpd_req_t *req) {
  httpd_resp_set_type(req, "text/html");

  return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

// =====================================================
// HTTP: STATUS JSON
// =====================================================

static esp_err_t status_handler(httpd_req_t *req) {
  RecognitionResult result;

  xSemaphoreTake(data_mutex, portMAX_DELAY);

  result = latest_result;

  xSemaphoreGive(data_mutex);

  char json[160];

  snprintf(json, sizeof(json),
           "{\"character\":\"%c\","
           "\"confidence\":%.1f,"
           "\"latency\":%lld,"
           "\"valid\":%s}",
           result.character, result.confidence, (long long)result.latency_ms,
           result.valid ? "true" : "false");

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");

  return httpd_resp_send(req, json, HTTPD_RESP_USE_STRLEN);
}

// =====================================================
// HTTP: MJPEG STREAM
// =====================================================

static esp_err_t stream_handler(httpd_req_t *req) {
  static const char *BOUNDARY = "\r\n--frame\r\n";

  httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=frame");

  httpd_resp_set_hdr(req, "Cache-Control",
                     "no-cache, no-store, must-revalidate");

  // Buffer riêng cho từng kết nối stream
  uint8_t *gray_copy = (uint8_t *)malloc(FRAME_SIZE);

  if (gray_copy == nullptr) {
    return ESP_ERR_NO_MEM;
  }

  uint8_t *jpg_buffer = nullptr;
  size_t jpg_length = 0;

  char header[96];

  while (true) {

    // Sao chép frame mới nhất
    xSemaphoreTake(data_mutex, portMAX_DELAY);

    memcpy(gray_copy, stream_frame, FRAME_SIZE);

    xSemaphoreGive(data_mutex);

    // Chuyển grayscale 96x96 thành JPEG
    ESP_LOGI(TAG, "Heap free: %u bytes | Largest block: %u bytes",
             (unsigned int)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned int)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    ESP_LOGI(TAG, "PSRAM free: %u bytes",
             (unsigned int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    bool converted =
        fmt2jpg(gray_copy, FRAME_SIZE, FRAME_WIDTH, FRAME_HEIGHT,
                PIXFORMAT_GRAYSCALE, STREAM_QUALITY, &jpg_buffer, &jpg_length);

    if (!converted || jpg_buffer == nullptr) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    // Gửi boundary
    esp_err_t ret = httpd_resp_send_chunk(req, BOUNDARY, strlen(BOUNDARY));

    if (ret != ESP_OK) {
      free(jpg_buffer);
      jpg_buffer = nullptr;
      break;
    }

    // Gửi header JPEG
    int header_length = snprintf(header, sizeof(header),
                                 "Content-Type: image/jpeg\r\n"
                                 "Content-Length: %u\r\n\r\n",
                                 (unsigned int)jpg_length);

    ret = httpd_resp_send_chunk(req, header, header_length);

    if (ret == ESP_OK) {
      ret = httpd_resp_send_chunk(req, (const char *)jpg_buffer, jpg_length);
    }

    free(jpg_buffer);
    jpg_buffer = nullptr;

    if (ret != ESP_OK) {
      ESP_LOGI(TAG, "Stream client disconnected");
      break;
    }

    // Giới hạn tốc độ stream
    vTaskDelay(pdMS_TO_TICKS(100));
  }

  free(gray_copy);

  return ESP_OK;
}

// =====================================================
// HTTP SERVER
// =====================================================

static httpd_handle_t start_webserver(void) {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();

  config.server_port = 80;
  config.max_uri_handlers = 3;
  config.stack_size = 8192;
  config.max_open_sockets = 4;

  httpd_handle_t server = nullptr;

  if (httpd_start(&server, &config) != ESP_OK) {
    ESP_LOGE(TAG, "HTTP server start failed");
    return nullptr;
  }

  httpd_uri_t index_uri = {};
  index_uri.uri = "/";
  index_uri.method = HTTP_GET;
  index_uri.handler = index_handler;

  httpd_register_uri_handler(server, &index_uri);

  httpd_uri_t stream_uri = {};
  stream_uri.uri = "/stream";
  stream_uri.method = HTTP_GET;
  stream_uri.handler = stream_handler;

  httpd_register_uri_handler(server, &stream_uri);

  httpd_uri_t status_uri = {};
  status_uri.uri = "/status";
  status_uri.method = HTTP_GET;
  status_uri.handler = status_handler;

  httpd_register_uri_handler(server, &status_uri);

  ESP_LOGI(TAG, "HTTP server started");

  return server;
}

// =====================================================
// UPDATE STREAM FRAME
// =====================================================

static void update_stream_frame(camera_fb_t *fb) {
  if (fb == nullptr || fb->format != PIXFORMAT_GRAYSCALE ||
      fb->width != FRAME_WIDTH || fb->height != FRAME_HEIGHT ||
      fb->len < FRAME_SIZE) {
    return;
  }

  xSemaphoreTake(data_mutex, portMAX_DELAY);

  memcpy(stream_frame, fb->buf, FRAME_SIZE);

  xSemaphoreGive(data_mutex);
}

// =====================================================
// EMNIST INFERENCE
// Giữ nguyên logic tiền xử lý của bạn
// =====================================================

static void run_inference(camera_fb_t *fb) {
  int8_t *input_data = input->data.int8;

  int start_x = (96 - 84) / 2;
  int start_y = (96 - 84) / 2;

  int step = 84 / 28;

  for (int y = 0; y < 28; y++) {

    for (int x = 0; x < 28; x++) {

      int src_x = start_x + x * step;
      int src_y = start_y + y * step;

      uint8_t raw = fb->buf[src_y * 96 + src_x];

      // 1. Invert grayscale
      uint8_t inverted = 255 - raw;

      // 2. Noise filtering
      if (inverted < 40) {
        inverted = 0;
      }

      // 3. INT8 conversion
      input_data[y * 28 + x] = (int8_t)((int16_t)inverted - 128);
    }
  }

  // Inference
  int64_t start_time = esp_timer_get_time();

  if (interpreter->Invoke() == kTfLiteOk) {

    int64_t latency_ms = (esp_timer_get_time() - start_time) / 1000;

    const int8_t *out_data = output->data.int8;

    int best_idx = 0;
    int8_t max_score = out_data[0];

    for (int i = 1; i < 47; i++) {

      if (out_data[i] > max_score) {
        max_score = out_data[i];
        best_idx = i;
      }
    }

    float confidence = ((float)max_score + 128.0f) / 256.0f * 100.0f;

    // Update dashboard data
    xSemaphoreTake(data_mutex, portMAX_DELAY);

    latest_result.character = EMNIST_LABELS[best_idx];

    latest_result.confidence = confidence;
    latest_result.latency_ms = latency_ms;
    latest_result.valid = true;

    xSemaphoreGive(data_mutex);

    if (confidence > RESULT_THRESHOLD) {

      ESP_LOGI(TAG, "-> Ket qua: '%c' | Confidence: %.1f%% | Latency: %lld ms",
               EMNIST_LABELS[best_idx], confidence, (long long)latency_ms);
    }
  } else {
    ESP_LOGE(TAG, "TFLite Invoke failed!");
  }
}

// =====================================================
// MAIN APPLICATION
// =====================================================

extern "C" void app_main(void) {
  ESP_LOGI(TAG, "Starting ESP32-S3 EMNIST System");

  // Shared memory and mutex
  data_mutex = xSemaphoreCreateMutex();

  if (data_mutex == nullptr) {
    ESP_LOGE(TAG, "Mutex creation failed");
    return;
  }

  stream_frame = (uint8_t *)heap_caps_malloc(FRAME_SIZE, MALLOC_CAP_8BIT);

  if (stream_frame == nullptr) {
    ESP_LOGE(TAG, "Stream buffer allocation failed");
    return;
  }

  memset(stream_frame, 255, FRAME_SIZE);

  // 1. Camera
  if (init_camera() != ESP_OK) {
    ESP_LOGE(TAG, "Camera initialization failed");
    return;
  }

  ESP_LOGI(TAG, "OV2640 READY");

  // 2. TFLite Micro
  if (!init_tflm()) {
    ESP_LOGE(TAG, "TFLite initialization failed");

    while (true) {
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }

  ESP_LOGI(TAG, "EMNIST INT8 READY");

  // 3. Wi-Fi
  if (!init_wifi()) {
    ESP_LOGE(TAG, "WiFi unavailable. Inference will continue offline.");
  } else {
    // 4. HTTP server
    start_webserver();
  }

  ESP_LOGI(TAG, "Starting recognition loop");

  // 5. Main inference loop
  while (true) {

    camera_fb_t *fb = esp_camera_fb_get();

    if (fb == nullptr) {
      ESP_LOGE(TAG, "Camera capture failed");

      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    if (fb->format != PIXFORMAT_GRAYSCALE || fb->width != FRAME_WIDTH ||
        fb->height != FRAME_HEIGHT || fb->len < FRAME_SIZE) {

      ESP_LOGE(TAG, "Unexpected camera framebuffer format");

      esp_camera_fb_return(fb);

      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    // Copy image for web streaming
    update_stream_frame(fb);

    // Return framebuffer before inference
    esp_camera_fb_return(fb);

    // Capture has been copied, so inference uses the copy
    // to avoid holding camera framebuffer during Invoke().
    // The inference function below uses a temporary frame.
    uint8_t *inference_frame = (uint8_t *)malloc(FRAME_SIZE);

    if (inference_frame == nullptr) {
      ESP_LOGE(TAG, "Inference frame allocation failed");
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    xSemaphoreTake(data_mutex, portMAX_DELAY);

    memcpy(inference_frame, stream_frame, FRAME_SIZE);

    xSemaphoreGive(data_mutex);

    // Use a local framebuffer descriptor for unchanged preprocessing
    camera_fb_t inference_fb = {};
    inference_fb.buf = inference_frame;
    inference_fb.width = FRAME_WIDTH;
    inference_fb.height = FRAME_HEIGHT;
    inference_fb.len = FRAME_SIZE;
    inference_fb.format = PIXFORMAT_GRAYSCALE;

    run_inference(&inference_fb);

    free(inference_frame);

    vTaskDelay(pdMS_TO_TICKS(400));
  }
}
