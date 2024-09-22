import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import google.generativeai as genai
from PIL import Image
from werkzeug.utils import secure_filename
import os
import json
from fpdf import FPDF
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import List
import textwrap
from IPython.display import display, Markdown
from PIL import Image
import shutil
from werkzeug.utils import secure_filename
import urllib.parse
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sns.set_theme(color_codes=True)
uploaded_df = None
document_analyzed = False
question_responses = []


def format_text(text):
    # Replace **text** with <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace any remaining * with <br>
    text = text.replace('*', '<br>')
    return text

def clean_data(df):
    # Step 1: Clean currency-related columns
    for col in df.columns:
        if any(x in col.lower() for x in ['value', 'price', 'cost', 'amount']):
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '').str.replace('£', '').str.replace('€', '').replace('[^\d.-]', '', regex=True).astype(float)
    
    # Step 2: Drop columns with more than 25% missing values
    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > 0.25].index
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Step 3: Fill missing values for remaining columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
    
    # Step 4: Convert object-type columns to lowercase
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    # Step 5: Drop columns with only one unique value
    unique_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=unique_value_columns, inplace=True)

    return df




def clean_data2(df):
    for col in df.columns:
        if 'value' in col or 'price' in col or 'cost' in col or 'amount' in col or 'Value' in col or 'Price' in col or 'Cost' in col or 'Amount' in col:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '')
                df[col] = df[col].str.replace('£', '')
                df[col] = df[col].str.replace('€', '')
                df[col] = df[col].replace('[^\d.-]', '', regex=True).astype(float)

    null_percentage = df.isnull().sum() / len(df)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    return df



def generate_plot(df, plot_path, plot_type):
    df = clean_data(df)
    excluded_words = ["name", "postal", "date", "phone", "address", "code", "id"]

    if plot_type == 'countplot':
        cat_vars = [col for col in df.select_dtypes(include='object').columns 
                    if all(word not in col.lower() for word in excluded_words) and df[col].nunique() > 1]
        
        for col in cat_vars:
            if df[col].nunique() > 10:
                top_categories = df[col].value_counts().index[:10]
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')

        num_cols = len(cat_vars)
        num_rows = (num_cols + 1) // 2
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        for i, var in enumerate(cat_vars):
            category_counts = df[var].value_counts()
            top_values = category_counts.index[:10][::-1]
            filtered_df = df.copy()
            filtered_df[var] = pd.Categorical(filtered_df[var], categories=top_values, ordered=True)
            sns.countplot(x=var, data=filtered_df, order=top_values, ax=axs[i])
            axs[i].set_title(var)
            axs[i].tick_params(axis='x', rotation=30)
            
            total = len(filtered_df[var])
            for p in axs[i].patches:
                height = p.get_height()
                axs[i].annotate(f'{height/total:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

            sample_size = filtered_df.shape[0]
            

        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    elif plot_type == 'histplot':
        num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns
                    if all(word not in col.lower() for word in excluded_words)]
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=min(3, num_cols), figsize=(15, 5*num_rows))
        axs = axs.flatten()

        plot_index = 0

        for i, var in enumerate(num_vars):
            if len(df[var].unique()) == len(df):
                fig.delaxes(axs[plot_index])
            else:
                sns.histplot(df[var], ax=axs[plot_index], kde=True, stat="percent")
                axs[plot_index].set_title(var)
                axs[plot_index].set_xlabel('')

            sample_size = df.shape[0]
            

            plot_index += 1

        for i in range(plot_index, len(axs)):
            fig.delaxes(axs[i])

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process/", response_class=HTMLResponse)
async def process_file(request: Request, file: UploadFile = File(...)):
    global df, uploaded_file, document_analyzed, file_path, file_extension
    uploaded_file = file
    file_location = f"static/{file.filename}"
    
    # Save the uploaded file to the server
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension == '.csv':
        file_path = 'dataset.csv'
        df = pd.read_csv(file_location, delimiter=",")
        df.to_csv(file_path, index=False)  # Save as dataset.csv
    elif file_extension == '.xlsx':
        file_path = 'dataset.xlsx'
        df = pd.read_excel(file_location)
        df.to_excel(file_path, index=False)  # Save as dataset.xlsx
    else:
        raise HTTPException(status_code=415, detail="Unsupported file format")

    # Get columns of the DataFrame
    columns = df.columns.tolist()

    return templates.TemplateResponse("upload.html", {"request": request, "columns": columns})


@app.post("/result")
async def result(request: Request,  
                 target: str = Form(...),
                 algorithm: str = Form(...)):
    global df, api
    global plot1_path, plot2_path, plot3_path, plot4_path, plot5_path, plot6_path, plot7_path, plot8_path, plot9_path, plot10_path, plot11_path
    global response1, response2, response3, response4, response5, response6, response7, response8, response9, response10, response11


    api = "AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA"
    excluded_words = ["name", "postal", "date", "phone", "address", "id"]
    
    if df[target].dtype in ['float64', 'int64']:
        unique_values = df[target].nunique()

        # If unique values > 20, treat it as regression, else classification
        if unique_values > 20:
            method = "Regression"
        else:
            method = "Classification"
    else:
        # If the target is not numeric, treat it as classification
        method = "Classification"



    # Initialize response3 and plot3_path to None
    response3 = None
    plot3_path = None
    response4 = None
    plot4_path = None
    response6 = None
    plot6_path = None
    response8 = None  # Initialize response8
    plot8_path = None  # Initialize plot8_path
    response9 = None  # Initialize response9
    plot9_path = None  # Initialize plot9_path
    response10 = None  # Initialize response8
    plot10_path = None  # Initialize plot8_path
    response11 = None  # Initialize response9
    plot11_path = None  # Initialize plot9_path

    if method == "Classification":
        cat_vars = [col for col in df.select_dtypes(include=['object']).columns 
                        if all(word not in col.lower() for word in excluded_words)]

        # Exclude the target variable from the list if it exists in cat_vars
        if target in cat_vars:
            cat_vars.remove(target)

        # Create a figure with subplots, but only include the required number of subplots
        num_cols = len(cat_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a count plot for each categorical variable
        for i, var in enumerate(cat_vars):
            top_categories = df[var].value_counts().nlargest(5).index
            filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]  # Exclude rows with NaN values in the variable
                    
            # Replace less frequent categories with "Other" if there are more than 5 unique values
            if df[var].nunique() > 5:
                other_categories = df[var].value_counts().index[5:]
                filtered_df[var] = filtered_df[var].apply(lambda x: x if x in top_categories else 'Other')
                    
            sns.countplot(x=var, hue=target, stat="percent", data=filtered_df, ax=axs[i])
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
                    
            # Change y-axis label to represent percentage
            axs[i].set_ylabel('Percentage')

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        # Remove any remaining blank subplots
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot3_path = "static/multiclass_barplot.png"
        plt.savefig(plot3_path)
        plt.close(fig)

        #response 3
        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key=api)

        import PIL.Image

        img = PIL.Image.open("static/multiclass_barplot.png")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        #response = model.generate_content(img)
        response = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response.resolve()
        response3 = format_text(response.text)


    if method == "Classification":
        # Generate Multiclass Pairplot
        pairplot_fig = sns.pairplot(df, hue=target)
        plot6_path = "static/pair1.png"  # Use plot6_path
        pairplot_fig.savefig(plot6_path)  # Save the pairplot as a PNG file
        

        # Google Gemini Integration
        genai.configure(api_key=api)
        img = PIL.Image.open(plot6_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Generate response based on the pairplot
        response = model.generate_content([
            "You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image. Explain it by points.",
            img
        ])
        response.resolve()

        # Assign the response to response6
        response6 = format_text(response.text)

        # Include response6 and plot6_path in the data dictionary to be passed to the template
        

    if method == "Classification":
        # Multiclass Histplot
        # Get the names of all columns with data type 'object' (categorical columns)
        cat_cols = df.columns.tolist()

        # Get the names of all columns with data type 'int'
        int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
        int_vars = [col for col in int_vars if col != target]

        # Create a figure with subplots
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a histogram for each integer variable with hue='Attrition'
        for i, var in enumerate(int_vars):
            top_categories = df[var].value_counts().nlargest(10).index
            filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]
            sns.histplot(data=df, x=var, hue=target, kde=True, ax=axs[i], stat="percent")
            axs[i].set_title(var)

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

        # Adjust spacing between subplots
        fig.tight_layout()
        plt.xticks(rotation=45)
        plot4_path = "static/multiclass_histplot.png"
        plt.savefig(plot4_path)
        plt.close(fig)

        #response 4
        def to_markdown(text):
            text = text.replace('•', '  *')
            return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

        genai.configure(api_key=api)

        import PIL.Image

        img = PIL.Image.open("static/multiclass_histplot.png")
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response4 = model.generate_content(img)
        response4 = model.generate_content(["As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
        response4.resolve()
        response4 = format_text(response4.text)





    # Generate Pairplot
    pairplot_fig = sns.pairplot(df)
    plot5_path = "static/pair2.png"  
    pairplot_fig.savefig(plot5_path)  # Save the pairplot as a PNG file

    # Google Gemini Integration
    genai.configure(api_key=api)
    img = PIL.Image.open(plot5_path)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Generate response based on the pairplot
    response = model.generate_content([
        "You are a professional Data Analyst, write the complete conclusion and actionable insight based on the image. Explain it by points.",
        img
    ])
    response.resolve()

    # Assign the response to response5
    response5 = format_text(response.text)

    def generate_gemini_response(plot_path):
        
       
        genai.configure(api_key=api)
        img = Image.open(plot_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content([
            " As a marketing consultant, I want to understand consumer insights based on the chart and the market context so I can use the key findings to formulate actionable insights", 
            img
        ])
        response.resolve()
        return response.text

    plot1_path = generate_plot(df, 'static/plot1.png', 'countplot')
    plot2_path = generate_plot(df, 'static/plot2.png', 'histplot')

    response1 = format_text((generate_gemini_response(plot1_path)))
    response2 = format_text((generate_gemini_response(plot2_path)))

    from sklearn import preprocessing
    for col in df.select_dtypes(include=['object']).columns:
    
        # Initialize a LabelEncoder object
        label_encoder = preprocessing.LabelEncoder()
    
        # Fit the encoder to the unique values in the column
        label_encoder.fit(df[col].unique())
    
        # Transform the column using the encoder
        df[col] = label_encoder.transform(df[col])

    
    # Display Correlation Heatmap
    plot7_path = "static/correlation_matrix.png"
    fig, ax = plt.subplots(figsize=(30, 24))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.savefig(plot7_path)
    plt.close(fig)

    img = PIL.Image.open(plot7_path)
    response7 = format_text((generate_gemini_response(plot7_path)))





    X = df.drop(target, axis=1)
    y = df[target]
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

    from scipy import stats
    threshold = 3

    for col in X_train.columns:
        if X_train[col].nunique() > 20:
            # Calculate Z-scores for the column
            z_scores = np.abs(stats.zscore(X_train[col]))
            # Find and remove outliers based on the threshold
            outlier_indices = np.where(z_scores > threshold)[0]
            X_train = X_train.drop(X_train.index[outlier_indices])
            y_train = y_train.drop(y_train.index[outlier_indices])




    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics
    from sklearn.metrics import mean_absolute_percentage_error
    import math


    if algorithm == "Decision Tree":

        if method == "Regression":
            dtree = DecisionTreeRegressor()
            param_grid = {
                'max_depth': [4, 6, 8],
                'min_samples_split': [4, 6, 8],
                'min_samples_leaf': [1, 2, 3, 4],
                'random_state': [0, 42],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            dtree = DecisionTreeRegressor(**best_params)
            dtree.fit(X_train, y_train)

            y_pred = dtree.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            # Feature importance visualization
            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance (Decision Tree Regressor)', fontsize=18)
            plot8_path = "static/dtree_regressor.png"
            plt.savefig(plot8_path)
            img = PIL.Image.open(plot8_path)
            response8 = format_text((generate_gemini_response(plot8_path)))
            

        elif method == "Classification":
            dtree = DecisionTreeClassifier()
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'random_state': [0, 42]
            }
            grid_search = GridSearchCV(dtree, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            dtree = DecisionTreeClassifier(**best_params)
            dtree.fit(X_train, y_train)

            y_pred = dtree.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            prec = metrics.precision_score(y_test, y_pred, average='micro')
            recall = metrics.recall_score(y_test, y_pred, average='micro')

            # Feature importance visualization
            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": dtree.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance (Decision Tree Classifier)', fontsize=18)
            plot9_path = "static/dtree_classifier.png"
            plt.savefig(plot9_path)
            img = PIL.Image.open(plot9_path)
            response9 = format_text((generate_gemini_response(plot9_path)))

        

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier

    if algorithm == "Random Forest":

        if method == "Regression":
            rf = RandomForestRegressor()
            param_grid = {
                'max_depth': [4, 6, 8],                
                'random_state': [0, 42],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            rf = RandomForestRegressor(**best_params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            r2 = metrics.r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            # Feature importance visualization
            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rf.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance (Random Forest Regressor)', fontsize=18)
            plot10_path = "static/rf_regressor.png"
            plt.savefig(plot10_path)
            img = PIL.Image.open(plot10_path)
            response10 = format_text((generate_gemini_response(plot10_path)))

        elif method == "Classification":
            rf = RandomForestClassifier()
            param_grid = {               
                'max_depth': [3, 4, 5, 6],
                'random_state': [0, 42]
            }
            grid_search = GridSearchCV(rf, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            rf = RandomForestClassifier(**best_params)
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred, average='micro')
            prec = metrics.precision_score(y_test, y_pred, average='micro')
            recall = metrics.recall_score(y_test, y_pred, average='micro')

            # Feature importance visualization
            imp_df = pd.DataFrame({
                "Feature Name": X_train.columns,
                "Importance": rf.feature_importances_
            })
            fi = imp_df.sort_values(by="Importance", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=fi, x='Importance', y='Feature Name', ax=ax)
            ax.set_title('Top 10 Feature Importance (Random Forest Classifier)', fontsize=18)
            plot11_path = "static/rf_classifier.png"
            plt.savefig(plot11_path)
            img = PIL.Image.open(plot11_path)
            response11 = format_text((generate_gemini_response(plot11_path)))



    document_analyzed = True

    

    data = {
        "request": request,
        "response1": response1,
        "response2": response2,
        "response5": response5,
        "response7": response7,
        "plot1_path": plot1_path,
        "plot2_path": plot2_path,
        "plot5_path": plot5_path,
        "plot7_path": plot7_path,
        "show_conversation": document_analyzed,
        "question_responses": question_responses
    }

    # Conditionally include response3 and plot3_path if they exist
    if response3:
        data["response3"] = response3
    if plot3_path:
        data["plot3_path"] = plot3_path
    if response4:
        data["response4"] = response3
    if plot4_path:
        data["plot4_path"] = plot4_path
    if response6:
        data["response6"] = response6
    if plot6_path:
        data["plot6_path"] = plot6_path
    if response8:
        data["response8"] = response8
    if plot8_path:
        data["plot8_path"] = plot8_path
    if response9:
        data["response9"] = response9
    if plot9_path:
        data["plot9_path"] = plot9_path
    if response10:
        data["response10"] = response10
    if plot10_path:
        data["plot10_path"] = plot10_path
    if response11:
        data["response11"] = response11
    if plot11_path:
        data["plot11_path"] = plot11_path

    return templates.TemplateResponse("upload.html", data)




# Route for asking questions
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    global file_extension, question_responses, api
    global plot1_path, plot2_path, plot3_path, plot4_path, plot5_path, plot6_path, plot7_path, plot8_path, plot9_path, plot10_path, plot11_path
    global response1, response2, response3, response4, response5, response6, response7, response8, response9, response10, response11
    global document_analyzed

    # Check if a file has been uploaded
    if not file_extension:
        raise HTTPException(status_code=400, detail="No file has been uploaded yet.")

    # Initialize the LLM model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)

    # Determine the file extension and select the appropriate loader
    file_path = ''
    loader = None

    if file_extension.endswith('.csv'):
        file_path = 'dataset.csv'
        loader = UnstructuredCSVLoader(file_path, mode="elements")
    elif file_extension.endswith('.xlsx'):
        file_path = 'dataset.xlsx'
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Load and process the document
    try:
        docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

    # Combine document text
    text = "\n".join([doc.page_content for doc in docs])
    os.environ["GOOGLE_API_KEY"] = api

    # Initialize embeddings and create FAISS vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    document_search = FAISS.from_texts(chunks, embeddings)

    # Generate query embedding and perform similarity search
    query_embedding = embeddings.embed_query(question)
    results = document_search.similarity_search_by_vector(query_embedding, k=3)

    if results:
        retrieved_texts = " ".join([result.page_content for result in results])

        # Define the Summarize Chain for the question
        latest_response = "" if not question_responses else question_responses[-1][1]
        template1 = (
            f"{question} Answer the question based on the following:\n\"{text}\"\n:" +
            (f" Answer the Question with only 3 sentences. Latest conversation: {latest_response}" if latest_response else "")
        )
        prompt1 = PromptTemplate.from_template(template1)

        # Initialize the LLMChain with the prompt
        llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

        # Invoke the chain to get the summary
        try:
            response_chain = llm_chain1.invoke({"text": text})
            summary1 = response_chain["text"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error invoking LLMChain: {str(e)}")

        # Generate embeddings for the summary
        try:
            summary_embedding = embeddings.embed_query(summary1)
            document_search = FAISS.from_texts([summary1], embeddings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

        # Perform a search on the FAISS vector database
        try:
            if document_search:
                query_embedding = embeddings.embed_query(question)
                results = document_search.similarity_search_by_vector(query_embedding, k=1)

                if results:
                    current_response = format_text(results[0].page_content)
                else:
                    current_response = "No matching document found in the database."
            else:
                current_response = "Vector database not initialized."
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")
    else:
        current_response = "No relevant results found."

    # Append the question and response from FAISS search
    current_question = f"You asked: {question}"
    question_responses.append((current_question, current_response))

    # Save all results to output_summary.json
    save_to_json(question_responses)

    

    data = {
        "request": request,
        "response1": response1,
        "response2": response2,
        "response5": response5,
        "response7": response7,
        "plot1_path": plot1_path,
        "plot2_path": plot2_path,
        "plot5_path": plot5_path,
        "plot7_path": plot7_path,
        "show_conversation": True,
        "question_responses": question_responses
    }

    # Conditionally include response3 and plot3_path if they exist
    if response3:
        data["response3"] = response3
    if plot3_path:
        data["plot3_path"] = plot3_path
    if response4:
        data["response4"] = response3
    if plot4_path:
        data["plot4_path"] = plot4_path
    if response6:
        data["response6"] = response6
    if plot6_path:
        data["plot6_path"] = plot6_path
    if response8:
        data["response8"] = response8
    if plot8_path:
        data["plot8_path"] = plot8_path
    if response9:
        data["response9"] = response9
    if plot9_path:
        data["plot9_path"] = plot9_path
    if response10:
        data["response10"] = response10
    if plot10_path:
        data["plot10_path"] = plot10_path
    if response11:
        data["response11"] = response11
    if plot11_path:
        data["plot11_path"] = plot11_path

    return templates.TemplateResponse("upload.html", data)



def save_to_json(question_responses):
    outputs = {
        "question_responses": question_responses
    }
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
