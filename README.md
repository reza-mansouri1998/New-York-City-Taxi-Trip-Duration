# UrbanFlow AI: A Geospatial NYC Taxi Prediction Pipeline 🚕

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)
![LightGBM](https://img.shields.io/badge/LightGBM-Machine%20Learning-orange)
![Leaflet](https://img.shields.io/badge/Leaflet-Geospatial-lightgreen)

An end-to-end machine learning application designed to estimate travel times across New York City. Built with highly optimized ensemble tree models and interactive geospatial routing, this project bridges complex analytical systems with clean, user-centric design.

## 🌟 Key Features

* **Advanced Machine Learning:** Powered by a finely-tuned LightGBM regression model trained on millions of historical NYC taxi records.
* **Geospatial Intelligence:** Utilizes Leaflet.js, reverse geocoding via Nominatim, and road-network snapping via OSRM to seamlessly translate user text inputs into precise mathematical Haversine distances.
* **Modular MLOps Pipeline:** Architected with strict separation of concerns, featuring dedicated Python modules for data ingestion, transformation, and global model caching.
* **Interactive Analytics:** A fully responsive Chart.js dashboard providing exploratory data analysis, including feature importance and temporal trip patterns.

## 🏗️ System Architecture & Pipeline

*(Add your pipeline diagram image here)*
<!-- Example: ![ML Pipeline Architecture](docs/pipeline_diagram.png) -->

The backend architecture follows strict MLOps principles:
1. **Data Ingestion:** Reads and splits raw Kaggle taxi data.
2. **Data Transformation:** Engineers temporal features and calculates spatial distances (Haversine), scaling numerical inputs via `StandardScaler`.
3. **Model Training:** Trains and tunes the LightGBM Regressor.
4. **Flask API:** Serves the pre-trained model and preprocessor (`.pkl` files) in memory for millisecond response times.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/UrbanFlow-AI.git](https://github.com/YourUsername/UrbanFlow-AI.git)
   cd UrbanFlow-AI


   UrbanFlow-AI/
├── artifacts/              # Serialized model and preprocessor (.pkl)
├── src/
│   ├── components/         # Data ingestion, transformation, model trainer
│   └── pipeline/           # Predict pipeline and CustomData bridge
├── templates/              # UI suite: index, predict, analytics, about
├── app.py                  # Flask application routing
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

👨‍💻 Author

Reza Mansouri
Data Science & AI Engineering
[LinkedIn](https://www.linkedin.com/in/reza-mansouri-18aa71252/) | GitHub

