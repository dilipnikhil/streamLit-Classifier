import streamlit as st
import pandas as pd
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
input_data =st.container()
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@st.cache
def get_data():
    data = pd.read_csv("imdb.csv")
    categories = data["CategoryName"].unique()
    data =data.iloc[:200,:]
    return data,categories


with open ("vectorizer.pkl","rb") as f:
    vectoriser = joblib.load(f)


with header:
    st.title("Text Moderation")
    st.text("This small project is about NLP, Multi-Class Textual Classification to be precise.!!!")


with dataset:
    st.header("IMDB Movie Review Dataset")
    st.text("The dataset consists of  35,000 records with 5 different classes!!")
    data,categories=get_data()
    st.write((data.iloc[47:60,:]).reset_index())
    st.text("The categories are")
    st.write(categories)
    st.subheader("The distribution of the dataset")
    st.bar_chart(data["CategoryName"].value_counts())

with model_training:
    st.header("Let's train your model !!!")
    st.text("Choose the hyper parameters you desire and see how the performance changes!!")
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider("What should be the maximum depth of the model?", min_value=1,max_value =10,value=2, step=1)
    n_estimators = sel_col.selectbox("How many trees should there be??",options = [100,200,300,"No Limit"],index = 0)
    max_features = sel_col.selectbox("What is the max number of features you want the model to select??",options= [1,2,4,6,8,10],index=3)
    rand_forest = RandomForestClassifier(max_depth=max_depth,n_estimators =n_estimators, max_features=max_features)
    X = data["comment"]
    y=data["CategoryName"]
    y=LabelEncoder().fit_transform(y)
    features = vectoriser.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test= train_test_split(features,y,test_size=0.25,random_state=1)
    model = rand_forest.fit(X_train,y_train)
    joblib.dump(model,"model.pkl")
    
    prediction=model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    disp_col.subheader("The Accuracy of the model is: ")
    disp_col.write(accuracy)

with input_data:
    st.subheader("Please enter a sentence after you have trained your model to get the predictions from it")
    sel_col_input, displ_col_input = st.columns(2)
    input_sentence = sel_col_input.text_input("Your sentence:")
    with open("Text-Moderator.pkl","rb") as f:
        model_text = joblib.load(f)
    with open ("vectorizer.pkl","rb") as f:
        vectoriser = joblib.load(f)
    displ_col_input.text("The Prediction of the model is:")
    if input_sentence:
        prediction_output = (model_text.predict(vectoriser.transform([input_sentence])))
        prob = max((model_text.predict_proba(vectoriser.transform([input_sentence]))[0]))
    else:
        prediction_output = " "
        prob = " "
    displ_col_input.write(prediction_output)
    displ_col_input.text("The corresponding probabilty is:")
    displ_col_input.write(prob)