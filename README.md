WatchTower is a hybrid network intrusion detection system that combines signature-based flow detection with machine learning anomaly detection to provide comprehensive threat coverage in encrypted network environments. The system addresses two critical limitations of existing IDS: (1) high false positive rates in ML-based systems, and (2) ineffectiveness of traditional deep packet inspection in encrypted traffic.

The project consists of two independently functional components that integrate to enhance overall detection capabilities:

**Component 1:** Flow-based signature detection system that captures network packets, aggregates them into flows, extracts behavioral features, and applies rule-based detection for known attacks (port scans, brute force, DDoS, reconnaissance).

**Component 2:** ML-based anomaly detection system that trains supervised learning models on labeled flow data to detect novel attacks and behavioral anomalies that signature rules cannot identify.

**Integration:** Both systems analyze the same network flows and share results through a common database. Signature detections provide validation for ML predictions (reducing false positives), while ML predictions identify novel attacks that signatures miss (increasing detection coverage). Signature-labeled data optionally enhances ML training for network-specific accuracy.

This repository handles the signature component
