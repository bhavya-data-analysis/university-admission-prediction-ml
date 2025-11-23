# 🎓 University Admission Prediction (Machine Learning Project)

This project predicts a student's probability of getting admitted into a graduate program using machine learning.  
The final model reaches **~82% accuracy** and helps students understand how GRE, GPA, TOEFL, SOP/LOR, and research experience affect admission chances.

---

## 📌 Overview
This project analyzes academic & profile features and builds regression models to estimate the **Chance of Admit**.  
It includes data preprocessing, model training, model comparison, and generating prediction outputs.

---

## 🧠 Key Features
- Predict admission chances as a percentage  
- Compare ML models  
- Clean preprocessing pipeline  
- Feature normalization  
- Training/testing split  
- Model evaluation metrics  

---

## 📊 Input Features Used
- GRE Score  
- TOEFL Score  
- GPA  
- SOP Rating  
- LOR Rating  
- Research Experience  

---

## 🧪 Models Compared
| Model | Accuracy | Notes |
|-------|----------|-------|
| **Linear Regression** | **82%** | Best performance, simplest & interpretable |
| Random Forest | ~80% | Good but complex |
| Decision Tree | ~61% | Overfits easily |

---

## 📂 Project Structure

```
university-admission-prediction-ml/
├── data/
│   └── admission_data.csv
├── notebooks/
│   └── admission_prediction.ipynb
├── src/
│   └── utils.py
├── outputs/
│   ├── charts/
│   └── example_output.png
└── README.md
```
---

## 🛠 Tech Stack
- Python  
- Scikit-Learn  
- Pandas, NumPy  
- Matplotlib / Seaborn  
- Jupyter Notebook  

---

## 🚀 How to Run
1. Upload the dataset into `/data`  
2. Open the notebook in `/notebooks`  
3. Run the preprocessing & model cells  
4. Enter your profile values to get prediction output  

---

## 📈 Example Output

Chance of Admit: 62%

---

## 📌 Future Improvements
- Add NLP analysis for SOP/LOR  
- Add more profile features  
- Build a web app version  

---

## ✨ Summary
- Linear Regression gave the best results (~82%).  
- GPA, GRE, and Research Experience were the **highest-impact features**.  
- The model is simple, transparent, and practical for students.



