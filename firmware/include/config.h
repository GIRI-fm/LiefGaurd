/**
 * ============================================================================
 * LiefGaurd Firmware Configuration Header
 * Pin definitions, I2C addresses, sampling rates, and AI thresholds
 * ============================================================================
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// ESP32 Pin Definitions
// ============================================================================

// I2C Configuration (Sensor Bus)
#define I2C_SDA_PIN           21      // GPIO21 - I2C Data line (MPU-6050 & PPG)
#define I2C_SCL_PIN           22      // GPIO22 - I2C Clock line
#define I2C_FREQUENCY         400000  // 400 kHz standard mode

// Dual MPU-6050 6-Axis IMU Addresses
#define MPU6050_ADDR_PRIMARY  0x68    // Primary IMU (AD0 = LOW)
#define MPU6050_ADDR_SECONDARY 0x69   // Secondary IMU (AD0 = HIGH)

// Analog Sensors
#define PPG_PULSE_PIN         34      // GPIO34 - ADC1_CH6 (Pulse Sensor Input)
#define ADC_RESOLUTION        12      // 12-bit ADC resolution (4096 levels)
#define ADC_ATTENUATION       3       // 11dB attenuation (0-3.3V range)

// Haptic Feedback & Audio
#define HAPTIC_MOTOR_PIN      18      // GPIO18 - PWM output for ERM motor
#define HAPTIC_PWM_FREQ       500     // 500 Hz base frequency
#define HAPTIC_PWM_CHANNEL    0       // LEDC channel 0
#define HAPTIC_PWM_RESOLUTION 10      // 10-bit PWM (0-1023)

// Debug & Status LEDs
#define STATUS_LED_PIN        2       // GPIO2 - Status indicator LED
#define ERROR_LED_PIN         15      // GPIO15 - Error indicator LED

// UART Configuration (Serial Debug)
#define DEBUG_UART_TX         1       // GPIO1 - UART0 TX
#define DEBUG_UART_RX         3       // GPIO3 - UART0 RX
#define DEBUG_BAUD_RATE       115200  // Serial monitor baud rate

// ============================================================================
// Sensor Sampling & Fusion Configuration
// ============================================================================

// MPU-6050 Sampling Rates
#define MPU6050_SAMPLE_RATE   50      // Hz (20ms per sample)
#define MPU6050_ACCEL_RANGE   16      // g (±16g range)
#define MPU6050_GYRO_RANGE    2000    // °/s (±2000°/s range)
#define MPU6050_FIFO_SIZE     512     // Bytes

// PPG Pulse Sensor
#define PPG_SAMPLE_RATE       100     // Hz (10ms per sample)
#define PPG_ADC_SAMPLES       10      // Oversampling count

// Madgwick Fusion Algorithm
#define FUSION_SAMPLE_RATE    50.0f   // Hz (must match MPU sample rate)
#define FUSION_BETA           0.1f    // Algorithm gain parameter

// ============================================================================
// TensorFlow Lite Micro AI Model Configuration
// ============================================================================

// Model Input Specs (must match training)
#define TFLITE_INPUT_SHAPE_0  128     // Timesteps (sequence length)
#define TFLITE_INPUT_SHAPE_1  6       // Channels (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
#define TFLITE_INPUT_TYPE     kTfLiteFloat32
#define TFLITE_OUTPUT_TYPE    kTfLiteFloat32

// Model Inference Window
#define INFERENCE_WINDOW_MS   2560    // 128 samples × 20ms per sample
#define INFERENCE_STRIDE_MS   500     // 25 samples stride for sliding window
#define INFERENCE_MAX_TIME_MS 50      // Max inference latency budget (sub-50ms requirement)

// Fatigue Index & Confidence Thresholds
#define FATIGUE_THRESHOLD_HIGH 0.75   // Confidence threshold for high fatigue alert
#define FATIGUE_THRESHOLD_MED  0.55   // Confidence threshold for medium fatigue
#define FATIGUE_THRESHOLD_LOW  0.35   // Confidence threshold for low fatigue

// Haptic Alert Triggering
#define HAPTIC_ALERT_INTENSITY 200    // PWM duty cycle (0-255) for motor vibration
#define HAPTIC_ALERT_DURATION_MS 150  // Vibration pulse duration
#define HAPTIC_ALERT_COOLDOWN_MS 1000 // Minimum time between consecutive alerts

// ============================================================================
// Bluetooth LE (BLE) Configuration
// ============================================================================

#define BLE_DEVICE_NAME       "LiefGaurd-Wrist"
#define BLE_SERVICE_UUID      "12345678-1234-5678-1234-56789abcdef0"
#define BLE_CHAR_IMU_UUID     "11111111-1111-1111-1111-111111111111"  // IMU telemetry
#define BLE_CHAR_FATIGUE_UUID "22222222-2222-2222-2222-222222222222" // Fatigue index
#define BLE_CHAR_HR_UUID      "33333333-3333-3333-3333-333333333333"  // Heart rate
#define BLE_MTU_SIZE          512     // Maximum transmission unit (bytes)
#define BLE_NOTIFY_INTERVAL   100     // Telemetry notification interval (ms)

// ============================================================================
// FreeRTOS Task Configuration
// ============================================================================

// Core 0: Sensor Acquisition & AI Inference
#define CORE0_TASK_PRIORITY   3       // Higher priority (sensor real-time requirement)
#define CORE0_STACK_SIZE      8192    // Bytes (TFLite Micro needs ~6KB)

// Core 1: BLE Communication & Haptics
#define CORE1_TASK_PRIORITY   2       // Medium priority (event-driven)
#define CORE1_STACK_SIZE      4096    // Bytes

// Task tick frequency
#define RTOS_TICK_RATE_HZ     1000    // 1ms tick (1000 Hz)

// ============================================================================
// Battery & Power Management
// ============================================================================

#define BATTERY_ADC_PIN       35      // GPIO35 - ADC1_CH7 (Battery voltage)
#define BATTERY_VOLTAGE_DIV   0.5     // Voltage divider ratio (2:1)
#define BATTERY_CRITICAL_MV   3300    // Critical battery level (3.3V)
#define DEEP_SLEEP_TIMEOUT_S  600     // Deep sleep after 10 minutes idle

// ============================================================================
// Data Packet Structures
// ============================================================================

// Telemetry packet size (BLE payload)
#define TELEMETRY_PACKET_SIZE 20      // Bytes per BLE notification
#define TELEMETRY_FIELDS     5        // accelX, gyroZ, heartRate, fatigue, timestamp

// ============================================================================
// Debug & Logging Configuration
// ============================================================================

#define DEBUG_ENABLE          1       // Enable serial debug output
#define LOG_SENSOR_DATA       1       // Log raw sensor readings
#define LOG_INFERENCE_TIME    1       // Log inference latency
#define LOG_FATIGUE_SCORES    1       // Log fatigue predictions

// ============================================================================
// Safety & Limits
// ============================================================================

#define MAX_CONSECUTIVE_ALERTS 5      // Max haptic alerts before cooldown
#define ALERT_COOLDOWN_PERIOD  5000   // Cooldown period (5 seconds)
#define WATCHDOG_TIMEOUT_MS    30000  // Watchdog timer (30 seconds)

#endif // CONFIG_H
