# Recipe Recommendation

[Deployed Link](https://recipe-recommendation-keia.onrender.com/)

## Overview

Recipe Recommendation is a web-based application that suggests recipes based on user preferences and/or ingredients using machine learning and NLP techniques. It helps users discover new cooking ideas, reduce food waste, and get personalized recipe recommendations.

## Features

* Input ingredients or dietary preferences to get recommended recipes.
* Uses embeddings/NLP to match user input with recipes.
* Web interface for easy interaction.
* Preprocessed recipe dataset with embeddings for fast recommendation.

## Technologies Used

* Python
* Flask
* HTML/CSS
* Machine Learning / Natural Language Processing
* Pandas, NumPy

## Getting Started

### Prerequisites

* Python 3.x
* pip

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/giri521/Recipe-Recommendation.git
   ```
2. Navigate to the project directory:

   ```
   cd Recipe-Recommendation
   ```
3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
4. Run the Flask app:

   ```
   python app.py
   ```
5. Open your browser and visit `http://127.0.0.1:5000` or use the deployed link.

## Usage

1. Open the web app.
2. Enter ingredients or a description of what you want to cook.
3. Submit to get recommended recipes.
4. View recipe details including ingredients and steps.

## File Structure

* `app.py` - Flask application.
* `train_data.py` - Preprocessing and embedding generation script.
* `data.json` - Raw recipe dataset.
* `bert_embeddings.npy` - Precomputed embeddings.
* `data_with_embeddings.csv` - Combined dataset with embeddings.
* `templates/` - HTML templates.
* `requirements.txt` - Python dependencies.

## Contributing

Contributions are welcome! Fork the repository and submit pull requests.
