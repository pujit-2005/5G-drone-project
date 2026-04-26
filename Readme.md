# 🚁 5G-Enabled UAV Precision Agriculture System

An end-to-end deep learning and 5G-based system for autonomous crop monitoring and precision pesticide spraying using drones.

---

## 📌 Overview

This project presents a **closed-loop precision agriculture system** where UAVs collect aerial data, process it using deep learning models, and generate real-time actionable insights for targeted spraying.

The system integrates:
- UAV-based sensing
- Deep learning (Autoencoder + CNN)
- 5G edge computing (MEC)
- Autonomous actuation

---

## 🧠 Key Idea

Traditional farming applies pesticides uniformly → inefficient.

This system enables:
> **“Spray only where needed, when needed.”**

---

## ⚙️ System Architecture

### Pipeline:

1. **Data Acquisition**
   - UAV captures aerial imagery (RGB + multispectral)

2. **5G Communication**
   - Data transmitted using:
     - eMBB → high bandwidth data
     - URLLC → low latency control
     - mMTC → IoT sensors

3. **Edge Processing (MEC)**
   - Real-time inference (<20 ms latency)

4. **Data Preprocessing**
   - Stacked Denoising Autoencoder (SDAE)
   - Removes noise from aerial data

5. **AI Analysis**
   - CNN detects anomalies:
     - weeds
     - water stress
     - crop defects

6. **Actuation**
   - Generates prescription map
   - Drone performs Variable Rate Application (VRA)

---

## 🧪 Dataset

- **Agriculture-Vision 2020**
- ~94,986 aerial images
- Multi-channel (RGB + NIR)
- Real-world agricultural conditions

---

## 🤖 Machine Learning Approach

### 1. Denoising (SDAE)
- Handles noisy UAV data
- Improves feature quality

---

### 2. Model Comparison

| Model | Accuracy |
|------|--------|
| LSTM | 68.4% |
| CNN  | 96.2% |

### ❌ LSTM Issue
- Lost spatial information  
- Poor detection of patterns  

### ✅ CNN Advantage
- Captures spatial features  
- High accuracy and reliability  

---

## 📈 Results

- ✅ 96.2% classification accuracy (CNN)
- ✅ Real-time inference (<20 ms using 5G MEC)
- 🌱 ~30% reduction in pesticide usage
- 🌾 ~23% increase in crop yield (projected)

---

## 📁 Project Structure
