idf_component_register(SRCS "emnist_esp32s3.cpp"
                    INCLUDE_DIRS "."
                    REQUIRES esp-tflite-micro esp32-camera esp_timer esp_wifi esp_event esp_netif nvs_flash esp_http_server freertos)
