# PhishGuard AI: Advanced Email Phishing Detection System

## Overview
PhishGuard AI is an advanced email analysis system designed to identify and prevent phishing attempts using state-of-the-art artificial intelligence and natural language processing techniques. The system employs the DistilBERT architecture along with traditional machine learning approaches to provide accurate phishing detection capabilities.

## Features
- **Advanced NLP Analysis:** Utilizes DistilBERT for deep linguistic analysis and context understanding
- **Multi-Model Comparison:** Implements multiple machine learning algorithms including XGBoost, Random Forest, and Neural Networks
- **Intuitive GUI:** Modern interface with real-time analysis feedback and risk assessment
- **Batch Processing:** Support for analyzing multiple emails simultaneously
- **File Format Support:** Handles various email formats including .msg, .eml, and .txt files
- **Risk Classification:** Five-level risk assessment system with detailed recommendations
- **Analysis History:** Tracks and stores previous analysis results for reference
- **Export Capabilities:** Results can be exported to CSV format for further analysis


## Results
The system achieves high accuracy in phishing detection:
- DistilBERT Model: 97.62% F1-score
- XGBoost Model: 88.23% F1-score
- Other models show varying performance between 75-87% F1-score

## Technical Details
### Models Implemented:
- DistilBERT (Primary NLP model)
- XGBoost
- Random Forest
- Gradient Boosting
- Logistic Regression
- Neural Networks
- Naive Bayes
- Decision Trees
- KNN
- SVM

### Features Analyzed:
- Word count and distribution
- URL patterns
- Email address patterns
- Sentence structure
- Language complexity
- Sentiment analysis
- Named Entity Recognition
- Readability scores
- Special character usage
- Numerical character analysis
- IP address detection

## Development Team
- **Developers:** Eyal Ronen & Gal Erez
- **Academic Supervisor:** Dr. Arik Pharan
- **Institution:** Ruppin Academic Center, 2024

## Project Status
This is the final version submitted as part of our computer engineering degree project. The system is fully functional but is provided as-is for academic purposes.

## Note
This repository contains the complete implementation of the PhishGuard AI system. Due to the nature of the models and data used, running the system requires specific dependencies and model files that are not included in this repository. The code is provided for review and academic purposes.

## Acknowledgments
Special thanks to:
- Dr. Arik Pharan for project supervision and guidance
- Ruppin Academic Center for providing the academic framework
- The open-source community for various libraries and tools used in this project

© 2024 PhishGuard AI - All Rights Reserved
