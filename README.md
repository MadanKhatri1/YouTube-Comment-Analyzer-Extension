# YouTube Comment Analyzer Extension

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-green)](https://lightgbm.readthedocs.io/)
[![Flask](https://img.shields.io/badge/API-Flask-lightgrey)](https://flask.palletsprojects.com/)
[![DVC](https://img.shields.io/badge/MLOps-DVC-orange)](https://dvc.org/)

A production-ready Chrome extension that performs real-time sentiment analysis on YouTube video comments using machine learning. The system combines a LightGBM classification model with TF-IDF vectorization to analyze comment sentiments, providing visual insights through interactive charts, word clouds, and sentiment trends. Read the [blog](https://medium.com/@madankh00123/building-a-smart-youtube-comment-analyzer-a-full-stack-ml-chrome-extension-ae69fe9449d3) for an explanation of the code.

## 🌟 Key Features

### Chrome Extension Frontend
- **Real-Time Comment Fetching**: Automatically extracts up to 500 comments from any YouTube video using the YouTube Data API v3
- **Sentiment Analysis Dashboard**: Interactive popup displaying:
  - Total comments and unique commenters
  - Average comment length and sentiment score (0-10 scale)
  - Sentiment distribution pie chart (Positive, Neutral, Negative)
  - Word cloud visualization of popular terms
  - Sentiment trend analysis over time
  - Top 25 comments with individual sentiment classifications
- **Dark Theme UI**: Modern, professional interface optimized for YouTube's aesthetic

### Machine Learning Pipeline
- **LightGBM Classifier**: Multi-class sentiment classifier (Positive: 1, Neutral: 0, Negative: -1)
- **TF-IDF Vectorization**: Advanced text feature extraction with trigrams (1-3 n-grams) and 10,000 max features
- **Robust Preprocessing**: 
  - Text normalization and cleaning
  - Stopword removal (excluding sentiment-critical words like "not", "but", "however")
  - Lemmatization using WordNet
  - Regex-based noise reduction
- **Hyperparameter Optimization**: Pre-tuned parameters for optimal performance:
  - Learning rate: 0.09
  - Max depth: 20
  - N estimators: 367

### MLOps Infrastructure
- **DVC Pipeline Orchestration**: End-to-end reproducible ML pipeline with 5 stages:
  1. Data Ingestion (fetches Reddit sentiment dataset)
  2. Data Preprocessing (text cleaning and normalization)
  3. Model Building (TF-IDF + LightGBM training)
  4. Model Evaluation (metrics computation and visualization)
  5. Model Registration (MLflow integration)
- **MLflow Tracking**: Centralized experiment tracking with:
  - Parameter logging
  - Metric tracking (precision, recall, F1-score per class)
  - Model versioning and signature inference
  - Confusion matrix visualization
  - Artifact management
- **Version Control**: Git + DVC for data and model versioning

### Flask REST API
- **Prediction Endpoints**:
  - `/predict`: Batch sentiment prediction
  - `/predict_with_timestamps`: Sentiment prediction with temporal data
- **Visualization Endpoints**:
  - `/generate_chart`: Dynamic pie chart generation for sentiment distribution
  - `/generate_wordcloud`: Word cloud image generation from comments
  - `/generate_trend_graph`: Monthly sentiment trend analysis
- **CORS Enabled**: Seamless cross-origin communication with Chrome extension
- **Production-Ready**: Deployed with Docker containerization

### DevOps & Deployment
- **Docker Support**: Containerized Flask API with optimized Python 3.10-slim base image
- **Makefile Automation**: Simplified commands for data processing, training, and environment setup
- **Structured Logging**: Comprehensive logging for debugging and monitoring
- **Error Handling**: Robust exception handling throughout the pipeline

## 📁 Project Structure

```
YouTube-Comment-Analyzer-Extension/
├── Frontend_Youtube_comment_analysis_extension/  # Chrome Extension
│   ├── manifest.json                              # Extension configuration (Manifest V3)
│   ├── popup.html                                 # Popup UI with dark theme
│   └── popup.js                                   # Extension logic and API communication
│
├── flask_app/                                     # Flask REST API
│   ├── app.py                                     # Main Flask server with prediction & visualization endpoints
│   └── requirements.txt                           # Flask app dependencies
│
├── src/                                           # Source code for ML pipeline
│   ├── data/
│   │   ├── data_ingestion.py                     # Dataset fetching and train-test split
│   │   └── data_preprocessing.py                  # Text preprocessing and feature engineering
│   ├── model/
│   │   ├── model_building.py                      # LightGBM training with hyperparameters
│   │   ├── model_evaluation.py                    # Model evaluation and MLflow logging
│   │   └── register_model.py                      # Model registration to MLflow
│   ├── features/                                  # Feature engineering modules
│   └── visualization/                             # Visualization utilities
│
├── scripts/                                       # Testing and utility scripts
│   ├── test_flask_api.py                          # API endpoint testing
│   ├── test_model.py                              # Model inference testing
│   ├── test_model_performance.py                  # Performance benchmarking
│   ├── test_model_signature.py                    # MLflow signature validation
│   └── promote_model.py                           # Model promotion workflow
│
├── data/                                          # Data storage (gitignored, DVC tracked)
│   ├── raw/                                       # Original train/test datasets
│   ├── interim/                                   # Preprocessed data
│   └── processed/                                 # Final modeling datasets
│
├── models/                                        # Trained model artifacts
│   ├── lgbm_model.pkl                             # Serialized LightGBM model
│   └── tfidf_vectorizer.pkl                       # Fitted TF-IDF vectorizer
│
├── dvc.yaml                                       # DVC pipeline definition
├── params.yaml                                    # Hyperparameters and configuration
├── Dockerfile                                     # Docker container specification
├── Makefile                                       # Automation commands
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.10 or higher
- **Chrome Browser**: For extension installation
- **Docker** (optional): For containerized deployment
- **DVC** (optional): For pipeline reproduction
- **MLflow Server** (optional): For experiment tracking

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/YouTube-Comment-Analyzer-Extension.git
cd YouTube-Comment-Analyzer-Extension
```

#### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

#### 4. Run the DVC Pipeline (Optional - for model training)

```bash
# Initialize DVC
dvc pull  # Pull data from remote storage (if configured)

# Run entire pipeline
dvc repro

# Or run individual stages
dvc repro data_ingestion
dvc repro data_preprocessing
dvc repro model_building
dvc repro model_evaluation
```

#### 5. Start the Flask API

**Option A: Local Execution**
```bash
cd flask_app
python app.py
# API will run on http://localhost:5000
```

**Option B: Docker Deployment**
```bash
# Build Docker image
docker build -t youtube-sentiment-api .

# Run container
docker run -p 5000:5000 youtube-sentiment-api
```

#### 6. Install Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **Load unpacked**
4. Select the `Frontend_Youtube_comment_analysis_extension/` folder
5. The extension icon should appear in your browser toolbar

### Configuration

#### YouTube API Key
Update the API key in `Frontend_Youtube_comment_analysis_extension/popup.js`:

```javascript
const API_KEY = 'YOUR_YOUTUBE_API_KEY_HERE';
```

Get your API key from [Google Cloud Console](https://console.cloud.google.com/apis/credentials).

#### MLflow Tracking Server
Update the tracking URI in `src/model/model_evaluation.py`:

```python
mlflow.set_tracking_uri("YOUR_MLFLOW_TRACKING_URI")
```

## 🎯 Usage

### Using the Chrome Extension

1. Navigate to any YouTube video (e.g., `https://www.youtube.com/watch?v=VIDEO_ID`)
2. Click the **YouTube Sentiment Analysis** extension icon
3. The extension will automatically:
   - Extract the video ID
   - Fetch up to 500 comments
   - Send comments to the Flask API for sentiment prediction
   - Display comprehensive sentiment analysis with visualizations

### API Endpoints

#### POST `/predict`
Perform sentiment analysis on a batch of comments.

**Request:**
```json
{
  "comments": ["Great video!", "This is terrible", "Okay content"]
}
```

**Response:**
```json
[
  {"comment": "Great video!", "sentiment": "1"},
  {"comment": "This is terrible", "sentiment": "-1"},
  {"comment": "Okay content", "sentiment": "0"}
]
```

#### POST `/predict_with_timestamps`
Sentiment analysis with timestamp preservation.

**Request:**
```json
{
  "comments": [
    {"text": "Amazing!", "timestamp": "2024-01-15T10:30:00Z", "authorId": "user123"}
  ]
}
```

**Response:**
```json
[
  {
    "comment": "Amazing!",
    "sentiment": "1",
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

#### POST `/generate_chart`
Generate a pie chart image for sentiment distribution.

**Request:**
```json
{
  "sentiment_counts": {"1": 150, "0": 80, "-1": 20}
}
```

**Response:** PNG image blob

#### POST `/generate_wordcloud`
Generate a word cloud from comments.

**Request:**
```json
{
  "comments": ["comment1", "comment2", ...]
}
```

**Response:** PNG image blob

#### POST `/generate_trend_graph`
Generate a monthly sentiment trend graph.

**Request:**
```json
{
  "sentiment_data": [
    {"timestamp": "2024-01-15T10:30:00Z", "sentiment": 1}
  ]
}
```

**Response:** PNG image blob

## 🧪 Testing

```bash
# Test Flask API endpoints
python scripts/test_flask_api.py

# Test model predictions
python scripts/test_model.py

# Benchmark model performance
python scripts/test_model_performance.py

# Validate MLflow model signature
python scripts/test_model_signature.py
```

## 📊 Model Performance

The LightGBM classifier is trained on a Reddit sentiment dataset and achieves:

- **Multi-class Classification**: 3 classes (Positive, Neutral, Negative)
- **Class Weighting**: Balanced class weights to handle imbalanced data
- **Regularization**: L1 (0.1) and L2 (0.1) regularization to prevent overfitting
- **Feature Engineering**: TF-IDF with trigrams captures contextual sentiment nuances

See `confusion_matrix_Test Data.png` for detailed performance metrics.

## 🔧 Development

### Project Organization

This project follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) structure for maintainability and reproducibility.

### Makefile Commands

```bash
make requirements       # Install Python dependencies
make data              # Generate dataset
make clean             # Remove compiled Python files
make lint              # Run flake8 linting
make create_environment # Create virtual environment
```

### DVC Pipeline Stages

1. **data_ingestion**: Fetches Reddit sentiment dataset and performs 80/20 train-test split
2. **data_preprocessing**: Applies text normalization, stopword removal, and lemmatization
3. **model_building**: Trains LightGBM model with TF-IDF features
4. **model_evaluation**: Computes metrics and logs to MLflow
5. **model_registration**: Registers model to MLflow Model Registry

## 🛠️ Technology Stack

### Backend
- **Python 3.10**: Core programming language
- **Flask**: REST API framework
- **LightGBM**: Gradient boosting classifier
- **Scikit-learn**: TF-IDF vectorization and metrics
- **NLTK**: Natural language preprocessing
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **WordCloud**: Word cloud generation

### MLOps
- **DVC**: Data and pipeline versioning
- **MLflow**: Experiment tracking and model registry
- **Docker**: Containerization
- **Make**: Build automation

### Frontend
- **Chrome Extension Manifest V3**: Modern extension framework
- **JavaScript (ES6+)**: Extension logic
- **YouTube Data API v3**: Comment fetching
- **HTML5/CSS3**: Dark-themed UI

## 📈 MLflow Integration

The project uses MLflow for comprehensive experiment tracking:

- **Experiment**: `dvc-pipeline-runs`
- **Logged Parameters**: All hyperparameters from `params.yaml`
- **Logged Metrics**: Precision, recall, F1-score per class
- **Logged Artifacts**: 
  - Confusion matrix images
  - TF-IDF vectorizer
  - Trained model with signature
- **Model Signature**: Input/output schema for deployment
- **Tags**: `model_type: LightGBM`, `task: Sentiment Analysis`, `dataset: YouTube Comments`

## 🐳 Docker Deployment

The Flask API is containerized for easy deployment:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgomp1
COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/
COPY lgbm_model.pkl /app/
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🔒 Security Considerations

- **API Key Security**: Store YouTube API keys securely (use environment variables in production)
- **CORS Configuration**: Currently allows all origins; restrict in production
- **Rate Limiting**: Implement rate limiting for API endpoints in production
- **Input Validation**: All API endpoints validate input before processing

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Reddit Sentiment Analysis Dataset from [Himanshu-1703/reddit-sentiment-analysis](https://github.com/Himanshu-1703/reddit-sentiment-analysis)
- **Project Template**: Based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- **ML Framework**: [LightGBM](https://lightgbm.readthedocs.io/) for efficient gradient boosting
- **YouTube API**: [Google YouTube Data API v3](https://developers.google.com/youtube/v3)

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the project maintainer.

---

**Built with ❤️ for better YouTube comment insights**
