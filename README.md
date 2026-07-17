# 🛠️ Predictive Maintenance for Milling Machines:

An industrial IoT and data science project designed to predict machine failures before they occur, reducing unplanned downtime and manufacturing costs. This system analyzes milling machine sensor telemetry to perform multi-class failure classification.

---

## 📊 Project Specifications:

* **Domain:** Industrial IoT / Manufacturing.
* **Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).
* **Input Features (Sensors):**
  * Air Temperature
  * Process Temperature
  * Rotational Speed
  * Torque
  * Tool Wear
* **Classification Task:** Multi-class Classification targeting specific failure types:
  * No Failure
  * Heat Dissipation Failure
  * Power Failure
  * Overstrain Failure
  * Tool Wear Failure

---

## ⚙️ Tech Stack & Workflow:

* **Data Exploration & Modeling:** Data cleaning, feature engineering (e.g., mechanical power), feature scaling (Standardization), and evaluating machine learning classifiers (such as Random Forest or Gradient Boosting).
* **Software Architecture & Deployment:** Refactoring predictive pipelines into modular Object-Oriented Programming (OOP) classes and deploying an interactive local web dashboard via **Streamlit**.
* **Version Control:** Managed collaboratively using Git & GitHub to streamline team integration.
