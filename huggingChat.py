import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 


import pandas as pd
from model import PredictReview

#Get Sentiment
def get_sentiment(text):
    #importing dataset
    data = pd.read_csv("output.csv")
    data['label_num'] = pd.get_dummies(data['label'],drop_first=True)

    review_predictor = PredictReview()

    model,coverter = review_predictor.base(data)

    # text = input("Enter the review: ")

    answer = review_predictor.test_sample(text,coverter,model)

    return answer

# load the Environment Variables. 
load_dotenv()
st.set_page_config(page_title="Amazon Product App")



# Sidebar contents
st.sidebar.image("WebHelpers.png")
st.divider()
st.sidebar.markdown('''
## About
This app is an Review Sentiment Analysis and a LLM-powered chatbot for Amazon Product related queries:
''')
st.title('Amazon Product Queries App 💬')

#menu = ['Amazon Review Sentiment Analysis','Product Queries BOT'] 
#choice  = st.selectbox("Select an option", menu)
#add_vertical_space(10)
#st.write('Made by [Harshil Agrawal](https://github.com/Harshil-Agrawal)')

custom_css = """
<style>
body {
    background-color: #f0f0f0; /* Set your desired background color here */
}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

st.header("Your Amazon Assistant 💬")
st.divider()

#tab1.write("this is tab 1")
#tab2.write("this is tab 2")
def main():
    tab1, tab2 = st.tabs(["Amazon Review Sentiment Analysis", "Product Queries BOT"])

    with tab1:

        st.subheader("Amazon Review Sentiment Analysis")
        with st.form(key='my_form'):
            raw_text = st.text_area("Enter the amazon review here:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.info("Sentiment:")
            answer = get_sentiment(raw_text)
            st.write(answer)
        
        # st.divider()

    
    with tab2:
        st.subheader("Product Queries BOT")    
        # Generate empty lists for generated and user.
        ## Assistant Response
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hi, please ask your queries related to amazon products?"]

        ## user question
        if 'user' not in st.session_state:
            st.session_state['user'] = ['Hi!']

        # Layout of input/response containers
        response_container = st.container()
        colored_header(label='', description='', color_name='yellow-30')
        input_container = st.container()

        # get user input
        def get_text():
            input_text = st.text_input("You: ", "", key="input")
            return input_text

        ## Applying the user input box
        with input_container:
            user_input = get_text()

        def chain_setup():
    
            #Template for the bot 
            template = """Your are amazon product related query bot so answer the queries related to only product related questions.\
            Also do not repeat any other question and only give the answer. Format the output and stop the generation of text after completion of 1 sentence and 1 line. End with a "." and Provide a complete answer for the Question : {question}
            Answer: """

            #Used ChatPrompt Template to set up a prompt for a chatbot
            # prompt_template = ChatPromptTemplate.from_template(template,partial_variables = {"format_instructions":format_instructions})
            prompt_template = ChatPromptTemplate.from_template(template)

            
            llm=HuggingFaceHub(huggingfacehub_api_token = "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",
                            repo_id="mistralai/Mistral-7B-v0.1",
                            model_kwargs={"max_new_tokens":100})
        
            llm_chain=LLMChain(
                llm=llm,
                prompt=prompt_template
            )

            return llm_chain


        # generate response
        def generate_response(question, llm_chain):
            response = llm_chain.run(question)
            return response

        ## load LLM
        llm_chain = chain_setup()

        # main loop
        with response_container:
            if user_input:
                response = generate_response(user_input, llm_chain)
                response = response.split("\n")
                if response[0][0].isdigit():
                    response[0] = response[0][2:]
                response = response[0].split(". ")
                response = '.'.join(response[:3])
                st.session_state.user.append(user_input)
                st.session_state.generated.append(response)
                
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

if __name__ == '__main__':
    main()
