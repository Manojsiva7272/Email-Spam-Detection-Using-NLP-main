
# 📧 E-mail Spam Detection Using Natural Language Processing (NLP)

## 📝 Project Overview
This project focuses on building a robust **E-mail Spam Detection** system using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The model accurately classifies emails as **spam** or **non-spam** based on the text content, helping users to filter out unwanted messages efficiently.

## 🔍 Problem Statement
With the increase in digital communication, spam emails pose a significant challenge for users and organizations. Manually filtering such emails is impractical. This project aims to automate the detection process using NLP and machine learning models.

## 🚀 Key Features
- **Data Preprocessing**: Text cleaning, stop word removal, and stemming to prepare raw email data.
- **Feature Extraction**: Implemented **TF-IDF Vectorization** to convert textual data into numerical format.
- **Model Building**: Used various machine learning models such as **Naive Bayes** and **Support Vector Machine (SVM)**.
- **Performance Evaluation**: Evaluated models using metrics like **Precision, Recall, F1-Score**, and a **Confusion Matrix**.

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`
- **NLP Techniques**: Text Preprocessing, TF-IDF Vectorization, Tokenization
- **Machine Learning Models**: Naive Bayes, Support Vector Machine (SVM), Logistic Regression

## 📊 Dataset
The dataset used consists of labeled emails marked as either "spam" or "ham" (non-spam). Each email is processed and fed into the model to determine whether it is spam or not.

## 💡 Workflow
1. **Data Cleaning**: Remove unnecessary characters, numbers, and punctuation.
2. **Text Normalization**: Convert to lowercase, remove stopwords, and apply stemming.
3. **Feature Extraction**: Generate feature vectors using **TF-IDF Vectorization**.
4. **Model Training**: Train the model using various machine learning algorithms.
5. **Model Evaluation**: Analyze results using **accuracy**, **precision**, **recall**, and **confusion matrix**.
6. **Prediction**: Classify new emails as spam or non-spam.

## 🏆 Results
The final model achieved an accuracy of **99%** on the test set, demonstrating its effectiveness in distinguishing between spam and non-spam emails.

### Confusion Matrix Summary:
- **True Positives (TP)**: Correctly classified spam emails.
- **True Negatives (TN)**: Correctly classified non-spam emails.
- **False Positives (FP)**: Non-spam emails incorrectly classified as spam.
- **False Negatives (FN)**: Spam emails incorrectly classified as non-spam.

## 📁 Project Structure
```
├── dataset/                     # Folder containing the dataset files
├── models/                      # Folder for trained models
├── README.md                    # Project documentation
├── spam_detection.ipynb         # Jupyter notebook with code and analysis
├── requirements.txt             # List of dependencies
└── results/                     # Output results and visualizations
```
## 🔧 Installation and Setup
1. Clone this repository:
   git clone https://github.com/Manojsiva7272/Email-Spam-Detection-NLP.git

2. Navigate to the project directory:
   cd Email-Spam-Detection-NLP
   
4. Install dependencies:
   pip install -r requirements.txt

5. Run the Jupyter notebook to explore the project:
   jupyter notebook spam_detection.ipynb

## 👨‍💻 How to Use
- Run the `spam_detection.ipynb` notebook.
- Follow through each cell to see the preprocessing, model building, and evaluation steps.
- Modify the code to test the model on custom email data.

## 🤝 Contributing
Contributions are welcome! If you have any suggestions or want to improve the model, feel free to open an issue or submit a pull request.

## 📧 Contact
For any questions or feedback, reach out to me at: **manojsiva202@gmail.com**
