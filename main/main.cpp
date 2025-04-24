// Author - Pavel Stepanov (xstepa77)
#include "model6.h"  // Generated Neuromodel
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include <math.h>
#include <esp_log.h>
#include <esp_timer.h>
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <inttypes.h> // For using PRIu32

#define TAG "ColorDetector"

// Sensor (TCS3200) pins
#define S0_PIN 25
#define S1_PIN 26
#define S2_PIN 27
#define S3_PIN 14
#define OUT_PIN 34

// This function replicates Arduino's pulseIn for ESP-IDF: it measures the duration
// of a pulse (HIGH or LOW) on the given pin, with a specified timeout.
unsigned long IRAM_ATTR pulseIn(uint8_t pin, uint8_t state, unsigned long timeout) {
    gpio_num_t gpio_num = (gpio_num_t)pin;
    uint8_t desired_state = (state == 0) ? 0 : 1;
    uint64_t startTime = esp_timer_get_time();

    // Wait until the pin stops being in the desired state (end of previous pulse)
    while (gpio_get_level(gpio_num) == desired_state) {
        if ((esp_timer_get_time() - startTime) > timeout) return 0;
    }

    // Wait for the pin to enter the desired state (start of new pulse)
    while (gpio_get_level(gpio_num) != desired_state) {
        if ((esp_timer_get_time() - startTime) > timeout) return 0;
    }

    uint64_t pulseStart = esp_timer_get_time();

    // Measure how long the pin stays in the desired state (the pulse duration)
    while (gpio_get_level(gpio_num) == desired_state) {
        if ((esp_timer_get_time() - startTime) > timeout) return 0;
    }

    return (unsigned long)(esp_timer_get_time() - pulseStart);
}

// Настройка фильтров для датчика
// These functions set the color filter on the sensor by controlling S2 and S3 pins
void setFilterRed() {
    gpio_set_level((gpio_num_t)S2_PIN, 0);
    gpio_set_level((gpio_num_t)S3_PIN, 0);
}

void setFilterGreen() {
    gpio_set_level((gpio_num_t)S2_PIN, 1);
    gpio_set_level((gpio_num_t)S3_PIN, 1);
}

void setFilterBlue() {
    gpio_set_level((gpio_num_t)S2_PIN, 0);
    gpio_set_level((gpio_num_t)S3_PIN, 1);
}

// Настройка модели TensorFlow Lite Micro
// Model and interpreter setup for TensorFlow Lite Micro
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 10 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Инициализация модели
// Initialize the TFLite model, interpreter, and tensors
void setupModel() {
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch!");
        while (true) { vTaskDelay(portMAX_DELAY); }
    }

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMul();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed!");
        while (true) { vTaskDelay(portMAX_DELAY); }
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI(TAG, "Model initialized.");
}

// Основная точка входа программы
extern "C" void app_main() {
    // Настройка GPIO (Configuring GPIO pins)
    gpio_config_t io_conf;
    // S0,S1,S2,S3 as outputs for controlling the sensor's frequency scaling and color filters
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = ((1ULL << S0_PIN) | (1ULL << S1_PIN) | (1ULL << S2_PIN) | (1ULL << S3_PIN));
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);

    // OUT_PIN as input to read the frequency pulses from the sensor
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << OUT_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);

    // Set S0 and S1 high to set the scaling frequency of the color sensor as per its datasheet
    gpio_set_level((gpio_num_t)S0_PIN, 1);
    gpio_set_level((gpio_num_t)S1_PIN, 1);

    setupModel();

    ESP_LOGI(TAG, "Starting color detection...");

    while (true) {
        uint64_t start = esp_timer_get_time();

        int redFrequency, greenFrequency, blueFrequency;

        // Measure frequencies for each color channel by setting the respective filters
        // and using pulseIn to measure the duration of pulses at OUT_PIN.
        setFilterRed();
        vTaskDelay(pdMS_TO_TICKS(100));
        redFrequency = pulseIn(OUT_PIN, 0, 1000000);

        setFilterGreen();
        vTaskDelay(pdMS_TO_TICKS(100));
        greenFrequency = pulseIn(OUT_PIN, 0, 1000000);

        setFilterBlue();
        vTaskDelay(pdMS_TO_TICKS(100));
        blueFrequency = pulseIn(OUT_PIN, 0, 1000000);

        // Avoid division by zero by ensuring frequency is at least 1
        if (redFrequency == 0) redFrequency = 1;
        if (greenFrequency == 0) greenFrequency = 1;
        if (blueFrequency == 0) blueFrequency = 1;

        // Normalize frequencies to the 0-1 range (arbitrary scaling factor 400.0f based on original code)
        float R = (float)redFrequency / 400.0f;
        float G = (float)greenFrequency / 400.0f;
        float B = (float)blueFrequency / 400.0f;

        // Compute additional features for the model input
        float RG_ratio = R / (G + 1e-6f);
        float R_minus_G = R - G;
        float Dominant_G = G;

        // Statistical normalization (scaling) of inputs
        static const float mean_values[6] = {
            204.5043755f,
            130.96420048f,
            204.86614956f,
            3.23668095f,
            73.54017502f,
            130.96420048f
        };

        static const float scale_values[6] = {
            142.31178788f,
            153.36651622f,
            142.31983469f,
            4.74252431f,
            145.19994231f,
            153.36651622f
        };

        // Applying normalization
        float R_val = (R - mean_values[0]) / scale_values[0];
        float G_val = (G - mean_values[1]) / scale_values[1];
        float B_val = (B - mean_values[2]) / scale_values[2];
        float RG_val = (RG_ratio - mean_values[3]) / scale_values[3];
        float RmG_val = (R_minus_G - mean_values[4]) / scale_values[4];
        float DomG_val = (Dominant_G - mean_values[5]) / scale_values[5];

        // Set input tensor data
        input->data.f[0] = R_val;
        input->data.f[1] = G_val;
        input->data.f[2] = B_val;
        input->data.f[3] = RG_val;
        input->data.f[4] = RmG_val;
        input->data.f[5] = DomG_val;

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed!");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        // Extracting predicted probabilities for each class from the output tensor
        float pBlue  = output->data.f[0];
        float pGreen = output->data.f[1];
        float pRed   = output->data.f[2];
        float pNone  = output->data.f[3];

        // Determine the class with the highest probability
        float maxVal = pBlue;
        const char* colorName = "Blue";
        if (pGreen > maxVal) { maxVal = pGreen; colorName = "Green"; }
        if (pNone > maxVal)  { maxVal = pNone;  colorName = "None"; }
        if (pRed > maxVal)   { maxVal = pRed;   colorName = "Red"; }

        uint64_t end = esp_timer_get_time();
        float cycle_time_ms = (end - start) / 1000.0f;

        // Log results
        ESP_LOGI(TAG, "Cycle time: %.2f ms", cycle_time_ms);
        ESP_LOGI(TAG, "RGB Frequencies: %.2f %.2f %.2f -> Detected: %s", R*400.0f, G*400.0f, B*400.0f, colorName);
        ESP_LOGI(TAG, "pBlue: %.6f pGreen: %.6f pNone: %.6f pRed: %.6f", pBlue, pGreen, pNone, pRed);

        // Memory info
        ESP_LOGI(TAG, "Free heap: %" PRIu32, esp_get_free_heap_size());
        ESP_LOGI(TAG, "Minimum free heap: %" PRIu32, esp_get_minimum_free_heap_size());

        // Delay before next measurement
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
