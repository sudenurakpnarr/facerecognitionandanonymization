
> ⚠️ **Note:**  
> Datasets, trained models, and output files are not included in this repository due to **privacy and size constraints**.  
> Instructions for obtaining or generating them are provided in `KURULUM_TALIMATLARI.md`.

---

## 🧪 Experimental Setup

- **Environment:** Indoor webcam-based testing
- **Hardware:** Standard consumer-grade laptop (no GPU)
- **Scenario:** Multiple individuals (registered & unregistered)
- **Input:** Real-time video stream
- **Evaluation Focus:** Accuracy, privacy, and latency

---

## 📊 Performance Results

| Metric | Result |
|------|-------|
| Face Detection Accuracy | **98.6%** |
| Recognition Accuracy | **97.9%** |
| False Acceptance Rate (FAR) | **1.4%** |
| False Rejection Rate (FRR) | **2.1%** |
| Average Processing Time | **28 ms/frame** |

✅ Real-time performance (~30 FPS)  
✅ Stable operation with multiple faces  
✅ Reliable anonymization of unknown individuals  

---

## 🔒 Privacy Mechanism

- **Known faces:**  
  - Bounding box + identity label  
  - No anonymization applied  

- **Unknown faces:**  
  - Gaussian blur applied to facial region  
  - Identity information visually removed  

This ensures **privacy-by-design** while maintaining system usability.

---

## ⚠️ Limitations

- Recognition accuracy decreases under:
  - Extreme lighting
  - Heavy occlusion
  - Sharp face angles
- Gaussian blurring may not fully resist advanced re-identification models
- Requires prior enrollment of authorized users

---



## 📄 License

This project was developed for **academic purposes only**.
