import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import openai
import matplotlib.pyplot as plt
from pandasai import PandasAI
import numpy as np


def initialize_openai():
    AZURE_OPENAI_KEY =  st.secrets["AZURE_OPENAI_KEY"]
    AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
    AZURE_ENGINE_NAME = st.secrets["AZURE_ENGINE_NAME"]

    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = "2023-05-15"
    openai.api_key = AZURE_OPENAI_KEY

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=AZURE_OPENAI_KEY,
        engine=AZURE_ENGINE_NAME,
        deployment_id=AZURE_OPENAI_ENDPOINT
    )

    return llm

pandas_ai = PandasAI(initialize_openai())

def run_langchain(llm, transcript):
    df = pd.DataFrame({'transcription': [transcript]})
    template = """You will act as an expert in pharmacology and medicine.
    Given a doctors notes you will extract all the medication mentioned, with respective dosage and frequency of intake.
    If no medication is found output: "No medication found in this registry."
    If medication is found create a markdown table with the medication information containing the following columns: Medication; Dosage; Frequency.
    Empty fields in the markdown table should be field with -"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=llm, prompt=chat_prompt)

    answer = chain.run(df.loc[0, 'transcription'])

    return answer

def process_csv_data(df, prompt):
    tmp_df = pandas_ai(df, prompt=prompt)

    return tmp_df

def plot_graph(df):

    df.plot(kind='line')
    plt.title('Extract')
    plt.xlabel('Patient ID')
    plt.ylabel('Y')
    return plt

def main():
    st.set_page_config(page_title="Medical Transcript Extraction", page_icon="👨‍⚕️")
    st.title("👨‍⚕️ Medical Transcript Extraction")

    transcript = st.text_area("Enter the doctor's notes transcript here:")

    if st.button("Extract Medication"):
        llm = initialize_openai()

        result = run_langchain(llm, transcript)

        st.markdown("### Result:")
        st.write(result)

    st.title('Patient Data Analysis')

    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

    prompt = st.text_input('What is the information that you want to know from the patients?', 'Which are the 5 heavyest patients?')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        processed_data = process_csv_data(df, prompt)

        fig = plot_graph(processed_data)
        st.pyplot(fig)

if __name__ == "__main__":
    main()