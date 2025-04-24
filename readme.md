# Machine Learning on Embedded Platforms: Color Recognition with ESP32 and TFLM

## How it Works (Demonstration)

You can see the project working in real experiments here:
[Project Demonstration Video](https://drive.google.com/file/d/1CD42iendyMvMEkGteHB_Ra5dmoi5mz6o/view?usp=sharing)

## Description

This project demonstrates the capabilities of machine learning (ML) on devices with limited computational resources. It focuses on implementing a color recognition system on the ESP32 platform using TensorFlow Lite for Microcontrollers (TFLM). The application aims to reliably recognize Red, Green, and Blue colors while efficiently utilizing the ESP32's resources (Flash and RAM) and providing fast inference suitable for real-time operation.

## Project Goals

1.  Design and train a neural network for color recognition on a PC (using Python + TensorFlow).
2.  Optimize the model and convert it to the TensorFlow Lite format.
3.  Implement code on the ESP32 to read values from a color sensor and pass them to the neural network.
4.  Measure performance: inference time, RAM, and Flash usage.
5.  Test the system with real data to verify accuracy.

## Hardware Requirements

* **Embedded Platform:** LilyGO TTGO Mini32 ESP32-WROVER-B
    * Dual-core processor (up to 240 MHz)
    * Integrated Wi-Fi and Bluetooth
    * 4 MB Flash and 520 KB RAM
* **Color Sensor:** TCS3200
* **Breadboard and Jumper Wires**

### Sensor Pinout (TCS3200 to ESP32)

| Sensor Pin | Function         | ESP32 Pin | Note             |
| :--------- | :--------------- | :-------- | :--------------- |
| SO         | Frequency Scaler | GPIO25    | Set scaler ratio |
| S1         | Frequency Scaler | GPIO26    | Set scaler ratio |
| S2         | Color Filter     | GPIO27    | Select filter    |
| S3         | Color Filter     | GPIO14    | Select filter    |
| OUT        | Output Signal    | GPIO34    | Light intensity  |
| VCC        | Power Supply     | 3.3V      | Power source     |
| GND        | Ground           | GND       | Common ground    |

*(See Figure 1 in the documentation for a visual setup)*

## Software Requirements

* **ESP-IDF:** Version 5.3
* **VS Code** (or preferred IDE for ESP-IDF)
* **TensorFlow/Keras** (for model training)
* **esp-tflite-micro library:** Version 1.3.2
* **Python** (for data preparation)

## Solution Overview

1.  **Data Collection:** RGB values were measured using the TCS3200 sensor under various conditions and saved.
2.  **Data Preparation:**
    * Frequency values were normalized to a 0-1 range.
    * Features were extracted, such as R/G ratio, R-G difference, and the Green component value.
    * The dataset was augmented with synthetically generated data (using Gaussian noise) to improve robustness, resulting in 5000 data entries for Red, Green, Blue, and None classes.
3.  **Model Training & Optimization:**
    * An initial 64-64-4 network achieved >99% accuracy but was too large for the ESP32.
    * The model was optimized to an 8-8-4 structure (8 input, 2 hidden layers with 8 neurons each, 4 output neurons for Red, Green, Blue, None).
    * The optimized model maintained >99% accuracy while significantly reducing size (10x smaller).
4.  **ESP32 Implementation:**
    * The optimized model was converted to TensorFlow Lite for Microcontrollers format.
    * The model was deployed on the ESP32 using the ESP-IDF framework and the `esp-tflite-micro` library.
    * The implementation involves configuring GPIOs, reading sensor data, normalizing/feature extraction, and running inference with the TFLM model.

## Performance Results

| Parameter        | Value      |
| :--------------- | :--------- |
| Free Heap Memory | 290736 B   |
| Time per Cycle   | 300.74 ms  |
| Model Size       | 17 kB      |

* **Accuracy:** >99% on test data.
* **Memory Efficiency:** The optimized model uses only 17 kB of memory.
* **Inference Speed:** Color detection takes approximately 300 ms per cycle.

## Sources & Links

* **Model Trainer & Collected Data:** [https://github.com/espatiel/Training_Model](https://github.com/espatiel/Training_Model)
* **TensorFlow Lite Micro:** [https://github.com/tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro)
* **TCS3200 Datasheet:** [Link to Datasheet](https://www.laskakit.cz/user/related_files/taos-tcs3200-datasheet.pdf)
* **LilyGO ESP32 Datasheet:** [Link to Datasheet](https://www.laskakit.cz/user/related_files/wch-jiangsu-qin-heng-ch9102f.pdf)