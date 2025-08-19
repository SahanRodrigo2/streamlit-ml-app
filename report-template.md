
Project Report

1. Dataset Description & Rationale
The Iris dataset contains 150 rows with 4 numerical features (sepal length, sepal width, petal length, petal width) and a categorical target (species). It is suitable for classification tasks and widely used as a beginner dataset

2. Data Preprocessing
- No missing values in Iris.
- Standardized numerical features with StandardScaler.
- Split into 80% train, 20% test.

3. Model Selection & Evaluation
- Tried Logistic Regression & Random Forest.
- Used 5-fold CV for accuracy.
- Random Forest had higher accuracy (e.g., 96%).
- Selected Random Forest as final model.

4. Streamlit App Design
- Page structure (Home, Explore, Visualizations, Predict, Performance)
- Widgets: sliders, selectboxes, number inputs.
- Error handling if model/data not available.
  
5. Deployment
- GitHub repo URL -  https://github.com/SahanRodrigo2/streamlit-ml-app.git
- Streamlit Cloud URL - https://app-ml-app-bve6cyqsfk35e9y9bgkys7.streamlit.app/
- Steps & challenges faced

6. Challenges Faced
- Setting up Python venv.
- Understanding Streamlit sidebar navigation.
- Deploying to Streamlit Cloud (dependency issues solved with requirements.txt).
  
7. Reflection
- Learned end-to-end ML workflow: dataset → preprocessing → training → evaluation → app → deployment.
- Improved skills in Streamlit, GitHub, and cloud deployment.
- Next time would try a larger dataset (Titanic/Wine Quality).

## 7. Screenshots
- Add key app screenshots (Home, Visualizations, Predict, Performance)
