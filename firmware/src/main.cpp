/**
 * ============================================================================
 * LiefGaurd Firmware - Main Entry Point
 * ESP32 FreeRTOS Dual-Core Implementation
 * Core 0: Sensor Acquisition & TinyML Inference (Real-time)
 * Core 1: BLE Telemetry & Haptic Feedback (Event-driven)
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include "config.h"

// TensorFlow Lite Micro headers (included via platformio.ini)
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "liefguard_model.h"  // Generated model header

// ============================================================================
// Global Variables & Data Structures
// ============================================================================

// I2C Wire instance
TwoWire I2C0 = TwoWire(0);

// Sensor data buffers (circular)
volatile float imuBuffer[BUFFER_SIZE][6];  // [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
volatile int bufferIndex = 0;
volatile float currentHeartRate = 0.0f;
volatile float currentFatigueIndex = 0.0f;

// Inter-core communication queue
QueueHandle_t telemetryQueue;
QueueHandle_t hapticAlertQueue;

// BLE server components
BLEServer* pServer = nullptr;
BLECharacteristic* pCharIMU = nullptr;
BLECharacteristic* pCharFatigue = nullptr;
BLECharacteristic* pCharHeartRate = nullptr;
bool deviceConnected = false;

// TensorFlow Lite Micro components
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Inference timing
unsigned long lastInferenceTime = 0;
unsigned long inferenceStartTime = 0;

// Haptic alert state machine
unsigned long lastHapticAlertTime = 0;
int consecutiveAlerts = 0;

// ============================================================================
// Telemetry Packet Structure
// ============================================================================

struct TelemetryPacket {
    uint32_t timestamp;
    float accelX;
    float gyroZ;
    float heartRate;
    float fatigueIndex;
};

struct HapticAlert {
    uint8_t intensity;      // 0-255 PWM duty cycle
    uint16_t durationMs;    // Vibration duration
    uint8_t pattern;        // 0=single, 1=double, 2=continuous
};

// ============================================================================
// MPU-6050 Sensor Interface
// ============================================================================

class MPU6050 {
public:
    uint8_t address;
    float accelScale, gyroScale;

    MPU6050(uint8_t addr) : address(addr), accelScale(9.81f / 2048.0f), gyroScale(1.0f / 131.0f) {}

    bool init() {
        // Write to PWR_MGMT_1 (0x6B): Wake up MPU, disable sleep
        I2C0.beginTransmission(address);
        I2C0.write(0x6B);
        I2C0.write(0x00);  // CLK_SEL = 0, SLEEP = 0
        if (I2C0.endTransmission() != 0) return false;

        // Set sample rate divider (SMPRT_DIV) to achieve 50Hz
        // Internal clock = 1000Hz, divider = 19 → 50Hz
        I2C0.beginTransmission(address);
        I2C0.write(0x19);  // SMPRT_DIV register
        I2C0.write(19);
        if (I2C0.endTransmission() != 0) return false;

        // Configure accelerometer range (±16g) → ACCEL_CONFIG = 0x18
        I2C0.beginTransmission(address);
        I2C0.write(0x1C);
        I2C0.write(0x18);  // FS_SEL = 3 (±16g)
        if (I2C0.endTransmission() != 0) return false;

        // Configure gyroscope range (±2000°/s) → GYRO_CONFIG = 0x18
        I2C0.beginTransmission(address);
        I2C0.write(0x1B);
        I2C0.write(0x18);  // FS_SEL = 3 (±2000°/s)
        if (I2C0.endTransmission() != 0) return false;

        return true;
    }

    bool readAccelGyro(float& ax, float& ay, float& az, float& gx, float& gy, float& gz) {
        uint8_t data[14];
        I2C0.beginTransmission(address);
        I2C0.write(0x3B);  // ACCEL_XOUT_H register
        if (I2C0.endTransmission() != 0) return false;

        int bytesRead = I2C0.requestFrom((int)address, 14);
        if (bytesRead != 14) return false;

        for (int i = 0; i < 14; i++) {
            data[i] = I2C0.read();
        }

        // Parse accelerometer (16-bit signed values)
        int16_t accelRaw[3];
        accelRaw[0] = (data[0] << 8) | data[1];  // X
        accelRaw[1] = (data[2] << 8) | data[3];  // Y
        accelRaw[2] = (data[4] << 8) | data[5];  // Z

        // Parse gyroscope (16-bit signed values)
        int16_t gyroRaw[3];
        gyroRaw[0] = (data[8] << 8) | data[9];   // X
        gyroRaw[1] = (data[10] << 8) | data[11]; // Y
        gyroRaw[2] = (data[12] << 8) | data[13]; // Z

        // Convert to physical units
        ax = accelRaw[0] * accelScale;
        ay = accelRaw[1] * accelScale;
        az = accelRaw[2] * accelScale;
        gx = gyroRaw[0] * gyroScale;
        gy = gyroRaw[1] * gyroScale;
        gz = gyroRaw[2] * gyroScale;

        return true;
    }
};

// Global MPU6050 instances
MPU6050 mpu_primary(MPU_ADDR_PRIMARY);
MPU6050 mpu_secondary(MPU_ADDR_SECONDARY);

// ============================================================================
// TensorFlow Lite Micro Inference Setup
// ============================================================================

void setupTFLiteMicro() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(liefguard_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter, "Model schema version mismatch");
        return;
    }

    static tflite::AllOpsResolver resolver;
    static uint8_t tensor_arena[16 * 1024];  // 16KB arena for inference

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena), error_reporter);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (DEBUG_ENABLE) {
        Serial.println("[TFLite] Model initialized successfully");
    }
}

// ============================================================================
// Inference Function - Runs on Core 0
// ============================================================================

float runInference(float imuData[BUFFER_SIZE][6]) {
    inferenceStartTime = micros();

    // Copy IMU data to TFLite input tensor (float32)
    if (input->type != kTfLiteFloat32) {
        if (DEBUG_ENABLE) Serial.println("[ERROR] Input tensor type mismatch");
        return 0.0f;
    }

    float* inputData = input->data.f;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        for (int j = 0; j < 6; j++) {
            inputData[i * 6 + j] = imuData[i][j];
        }
    }

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        if (DEBUG_ENABLE) Serial.println("[ERROR] Inference failed");
        return 0.0f;
    }

    // Extract output (probability of CNS fatigue)
    float* outputData = output->data.f;
    float fatigueConfidence = outputData[0];  // Sigmoid output [0.0, 1.0]

    // Log inference timing
    unsigned long inferenceTime = (micros() - inferenceStartTime) / 1000;  // Convert to ms
    if (LOG_INFERENCE_TIME) {
        Serial.print("[INFERENCE] Latency: ");
        Serial.print(inferenceTime);
        Serial.println(" ms");
    }

    return fatigueConfidence;
}

// ============================================================================
// Haptic Feedback Control
// ============================================================================

void triggerHapticAlert(uint8_t intensity, uint16_t durationMs) {
    // State machine: check cooldown period
    unsigned long timeSinceLastAlert = millis() - lastHapticAlertTime;

    if (timeSinceLastAlert < HAPTIC_ALERT_COOLDOWN_MS) {
        if (DEBUG_ENABLE) Serial.println("[HAPTIC] Cooldown active, skipping alert");
        return;
    }

    if (consecutiveAlerts >= MAX_CONSECUTIVE_ALERTS) {
        if (timeSinceLastAlert < ALERT_COOLDOWN_PERIOD) {
            if (DEBUG_ENABLE) Serial.println("[HAPTIC] Max alerts reached, entering cooldown");
            return;
        }
        consecutiveAlerts = 0;  // Reset counter
    }

    // Queue haptic alert to Core 1
    HapticAlert alert = {intensity, durationMs, 0};
    xQueueSend(hapticAlertQueue, &alert, 10);

    lastHapticAlertTime = millis();
    consecutiveAlerts++;

    if (DEBUG_ENABLE) {
        Serial.print("[HAPTIC] Alert queued - intensity: ");
        Serial.print(intensity);
        Serial.print(", duration: ");
        Serial.print(durationMs);
        Serial.println(" ms");
    }
}

void executeHapticVibration(uint8_t intensity, uint16_t durationMs) {
    // Set PWM duty cycle (0-255 → 0-100%)
    ledcWrite(HAPTIC_PWM_CHANNEL, intensity);
    digitalWrite(STATUS_LED_PIN, HIGH);

    // Vibrate for specified duration
    vTaskDelay(pdMS_TO_TICKS(durationMs));

    // Stop vibration
    ledcWrite(HAPTIC_PWM_CHANNEL, 0);
    digitalWrite(STATUS_LED_PIN, LOW);

    if (LOG_SENSOR_DATA) {
        Serial.println("[HAPTIC] Vibration sequence complete");
    }
}

// ============================================================================
// BLE Server Callbacks
// ============================================================================

class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        if (DEBUG_ENABLE) Serial.println("[BLE] Device connected");
    }

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        if (DEBUG_ENABLE) Serial.println("[BLE] Device disconnected");
    }
};

void setupBLE() {
    BLEDevice::init(BLE_DEVICE_NAME);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    // Create GATT Service
    BLEService* pService = pServer->createService(BLE_SERVICE_UUID);

    // Create Characteristics
    pCharIMU = pService->createCharacteristic(
        BLE_CHAR_IMU_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCharIMU->addDescriptor(new BLE2902());

    pCharFatigue = pService->createCharacteristic(
        BLE_CHAR_FATIGUE_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCharFatigue->addDescriptor(new BLE2902());

    pCharHeartRate = pService->createCharacteristic(
        BLE_CHAR_HR_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCharHeartRate->addDescriptor(new BLE2902());

    // Start service
    pService->start();

    // Start advertising
    BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(BLE_SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    BLEDevice::startAdvertising();

    if (DEBUG_ENABLE) Serial.println("[BLE] Server initialized and advertising");
}

// ============================================================================
// CORE 0: Sensor Acquisition & TinyML Inference Task
// ============================================================================

void sensorAndInferenceTask(void* parameter) {
    if (DEBUG_ENABLE) Serial.println("[CORE0] Task started on Core 0");

    unsigned long sampleIntervalMs = 1000 / SAMPLE_RATE_HZ;  // 20ms for 50Hz
    TickType_t xLastWakeTime = xTaskGetTickCount();

    while (1) {
        // Read dual MPU-6050 IMUs
        float ax_p, ay_p, az_p, gx_p, gy_p, gz_p;
        float ax_s, ay_s, az_s, gx_s, gy_s, gz_s;

        bool success_p = mpu_primary.readAccelGyro(ax_p, ay_p, az_p, gx_p, gy_p, gz_p);
        bool success_s = mpu_secondary.readAccelGyro(ax_s, ay_s, az_s, gx_s, gy_s, gz_s);

        if (success_p && success_s) {
            // Average measurements from both IMUs
            float ax_avg = (ax_p + ax_s) / 2.0f;
            float ay_avg = (ay_p + ay_s) / 2.0f;
            float az_avg = (az_p + az_s) / 2.0f;
            float gx_avg = (gx_p + gx_s) / 2.0f;
            float gy_avg = (gy_p + gy_s) / 2.0f;
            float gz_avg = (gz_p + gz_s) / 2.0f;

            // Store in circular buffer
            imuBuffer[bufferIndex][0] = ax_avg;
            imuBuffer[bufferIndex][1] = ay_avg;
            imuBuffer[bufferIndex][2] = az_avg;
            imuBuffer[bufferIndex][3] = gx_avg;
            imuBuffer[bufferIndex][4] = gy_avg;
            imuBuffer[bufferIndex][5] = gz_avg;

            bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;

            if (LOG_SENSOR_DATA && bufferIndex % 5 == 0) {
                Serial.print("[SENSOR] ax=");
                Serial.print(ax_avg);
                Serial.print(", gz=");
                Serial.println(gz_avg);
            }

            // Run inference when buffer is full
            if (bufferIndex == 0) {
                float fatigueConfidence = runInference((float(*)[6])imuBuffer);
                currentFatigueIndex = fatigueConfidence;

                if (LOG_FATIGUE_SCORES) {
                    Serial.print("[AI] Fatigue Index: ");
                    Serial.println(fatigueConfidence);
                }

                // Trigger haptic alert if fatigue exceeds threshold
                if (fatigueConfidence > FATIGUE_THRESHOLD) {
                    if (DEBUG_ENABLE) Serial.println("[ALERT] High fatigue detected!");
                    triggerHapticAlert(HAPTIC_ALERT_INTENSITY, HAPTIC_ALERT_DURATION_MS);
                }

                // Queue telemetry packet
                TelemetryPacket packet = {
                    (uint32_t)millis(),
                    ax_avg,
                    gz_avg,
                    currentHeartRate,
                    fatigueConfidence
                };
                xQueueSend(telemetryQueue, &packet, 10);
            }
        }

        // Delay to maintain 50Hz sampling rate
        vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(sampleIntervalMs));
    }
}

// ============================================================================
// CORE 1: BLE Communication & Haptic Feedback Task
// ============================================================================

void bleAndHapticTask(void* parameter) {
    if (DEBUG_ENABLE) Serial.println("[CORE1] Task started on Core 1");

    while (1) {
        // Process haptic alerts from queue
        HapticAlert alert;
        if (xQueueReceive(hapticAlertQueue, &alert, 0) == pdTRUE) {
            executeHapticVibration(alert.intensity, alert.durationMs);
        }

        // Broadcast telemetry via BLE
        if (deviceConnected) {
            TelemetryPacket packet;
            if (xQueueReceive(telemetryQueue, &packet, 0) == pdTRUE) {
                // Build BLE payload (20 bytes for MTU=23)
                uint8_t payload[20];
                payload[0] = (packet.timestamp >> 24) & 0xFF;
                payload[1] = (packet.timestamp >> 16) & 0xFF;
                payload[2] = (packet.timestamp >> 8) & 0xFF;
                payload[3] = packet.timestamp & 0xFF;

                memcpy(&payload[4], &packet.accelX, 4);
                memcpy(&payload[8], &packet.gyroZ, 4);
                memcpy(&payload[12], &packet.heartRate, 4);
                memcpy(&payload[16], &packet.fatigueIndex, 4);

                pCharIMU->setValue(payload, 20);
                pCharIMU->notify();
                pCharFatigue->setValue((uint8_t*)&packet.fatigueIndex, 4);
                pCharFatigue->notify();
            }
        }

        vTaskDelay(pdMS_TO_TICKS(BLE_NOTIFY_INTERVAL));
    }
}

// ============================================================================
// Setup Function (Runs on Core 1)
// ============================================================================

void setup() {
    Serial.begin(DEBUG_BAUD_RATE);
    delay(1000);
    Serial.println("\n[BOOT] LiefGaurd Firmware Initializing...");

    // Initialize I2C
    I2C0.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQUENCY);
    if (DEBUG_ENABLE) Serial.println("[I2C] Initialized at 400 kHz");

    // Initialize MPU-6050 sensors
    if (!mpu_primary.init() || !mpu_secondary.init()) {
        Serial.println("[ERROR] MPU-6050 initialization failed");
        while (1) delay(1000);
    }
    if (DEBUG_ENABLE) Serial.println("[SENSOR] Dual MPU-6050 initialized");

    // Initialize TensorFlow Lite Micro
    setupTFLiteMicro();

    // Initialize GPIO
    pinMode(HAPTIC_MOTOR_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    pinMode(ERROR_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);
    digitalWrite(ERROR_LED_PIN, LOW);

    // Configure PWM for haptic motor
    ledcSetup(HAPTIC_PWM_CHANNEL, HAPTIC_PWM_FREQ, HAPTIC_PWM_RESOLUTION);
    ledcAttachPin(HAPTIC_MOTOR_PIN, HAPTIC_PWM_CHANNEL);
    if (DEBUG_ENABLE) Serial.println("[HAPTIC] PWM initialized at 500 Hz");

    // Initialize BLE
    setupBLE();

    // Create FreeRTOS queues
    telemetryQueue = xQueueCreate(5, sizeof(TelemetryPacket));
    hapticAlertQueue = xQueueCreate(3, sizeof(HapticAlert));

    if (DEBUG_ENABLE) Serial.println("[QUEUE] Inter-core queues created");

    // Create Core 0 task (Sensor & Inference)
    xTaskCreatePinnedToCore(
        sensorAndInferenceTask,     // Task function
        "SensorInference",          // Task name
        CORE0_STACK_SIZE,           // Stack size
        nullptr,                    // Parameter
        CORE0_TASK_PRIORITY,        // Priority
        nullptr,                    // Task handle
        0);                         // Pinned to Core 0

    // Create Core 1 task (BLE & Haptics) - Already running on Core 1
    xTaskCreatePinnedToCore(
        bleAndHapticTask,
        "BLEHaptic",
        CORE1_STACK_SIZE,
        nullptr,
        CORE1_TASK_PRIORITY,
        nullptr,
        1);                         // Pinned to Core 1

    Serial.println("[BOOT] All systems initialized. Running inference loop...");
}

// ============================================================================
// Main Loop (Idle on Core 1)
// ============================================================================

void loop() {
    // FreeRTOS tasks handle all work
    vTaskDelay(pdMS_TO_TICKS(1000));

    // Periodic status update
    static unsigned long lastStatusTime = 0;
    if (millis() - lastStatusTime > 5000) {
        if (DEBUG_ENABLE) {
            Serial.print("[STATUS] Fatigue: ");
            Serial.print(currentFatigueIndex);
            Serial.print(", Connected: ");
            Serial.println(deviceConnected ? "YES" : "NO");
        }
        lastStatusTime = millis();
    }
}
