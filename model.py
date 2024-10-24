import streamlit as st
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


sec_key = "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"  
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"  

# Initializing the generative model from HuggingFace
llm_gen = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,
    temperature=0.7,
    token=sec_key
)

# Restaurant Name Generator Template
restaurant_name_template = '''
Generate a unique and creative name for a restaurant that serves {cuisine_type} cuisine.
Restaurant Name:
'''
prompt_restaurant_name = PromptTemplate(
    input_variables=['cuisine_type'],
    template=restaurant_name_template
)

# Menu Generator Template
menu_generator_template = '''
Generate a detailed menu with appetizer, main course, dessert, and drink options for a restaurant that serves {cuisine_type} cuisine.
Menu:
'''
prompt_menu_generator = PromptTemplate(
    input_variables=['cuisine_type'],
    template=menu_generator_template
)

# Special Dish Generator Template
special_dish_template = '''
Generate a signature dish for a restaurant that serves {cuisine_type} cuisine. Describe the dish in detail.
Signature Dish:
'''
prompt_special_dish = PromptTemplate(
    input_variables=['cuisine_type'],
    template=special_dish_template
)

# Restaurant Tagline Generator Template
tagline_template = '''
Generate a catchy and creative tagline for a restaurant named {restaurant_name} that serves {cuisine_type} cuisine.
Tagline:
'''
prompt_tagline = PromptTemplate(
    input_variables=['restaurant_name', 'cuisine_type'],
    template=tagline_template
)

# Cuisine Pairing Suggestion Template
cuisine_pairing_template = '''
Suggest another cuisine that pairs well with {cuisine_type} cuisine for a fusion restaurant.
Suggested Cuisine Pairing:
'''
prompt_cuisine_pairing = PromptTemplate(
    input_variables=['cuisine_type'],
    template=cuisine_pairing_template
)

st.title("Restaurant Creator - AI Assistant")

st.sidebar.title("Choose a Task")
task = st.sidebar.selectbox(
    "Task",
    (
        "Restaurant Name Generator",
        "Menu Generator",
        "Special Dish Generator",
        "Restaurant Tagline Generator",
        "Cuisine Pairing Suggestion"
    ),
)

if task == "Restaurant Name Generator":
    st.header("Restaurant Name Generator")
    cuisine_type = st.text_input("Enter the cuisine type (e.g., Italian, Indian, Chinese):")
    if st.button("Generate Restaurant Name"):
        if cuisine_type:
            restaurant_name_chain = LLMChain(llm=llm_gen, prompt=prompt_restaurant_name)
            response = restaurant_name_chain.run({"cuisine_type": cuisine_type})
            st.write("### Suggested Restaurant Name:")
            st.write(response)
        else:
            st.write("Please enter a cuisine type.")

# Menu Generator
elif task == "Menu Generator":
    st.header("Menu Generator")
    cuisine_type = st.text_input("Enter the cuisine type (e.g., Mexican, French, Japanese):")
    if st.button("Generate Menu"):
        if cuisine_type:
            menu_chain = LLMChain(llm=llm_gen, prompt=prompt_menu_generator)
            response = menu_chain.run({"cuisine_type": cuisine_type})
            st.write("### Suggested Menu:")
            st.write(response)
        else:
            st.write("Please enter a cuisine type.")

# Special Dish Generator
elif task == "Special Dish Generator":
    st.header("Special Dish Generator")
    cuisine_type = st.text_input("Enter the cuisine type (e.g., Mediterranean, Thai):")
    if st.button("Generate Signature Dish"):
        if cuisine_type:
            special_dish_chain = LLMChain(llm=llm_gen, prompt=prompt_special_dish)
            response = special_dish_chain.run({"cuisine_type": cuisine_type})
            st.write("### Signature Dish:")
            st.write(response)
        else:
            st.write("Please enter a cuisine type.")

# Restaurant Tagline Generator
elif task == "Restaurant Tagline Generator":
    st.header("Restaurant Tagline Generator")
    restaurant_name = st.text_input("Enter your restaurant name:")
    cuisine_type = st.text_input("Enter the cuisine type (e.g., Korean, Indian):")
    if st.button("Generate Tagline"):
        if restaurant_name and cuisine_type:
            tagline_chain = LLMChain(llm=llm_gen, prompt=prompt_tagline)
            response = tagline_chain.run({"restaurant_name": restaurant_name, "cuisine_type": cuisine_type})
            st.write("### Suggested Tagline:")
            st.write(response)
        else:
            st.write("Please enter both restaurant name and cuisine type.")

# Cuisine Pairing Suggestion
elif task == "Cuisine Pairing Suggestion":
    st.header("Cuisine Pairing Suggestion")
    cuisine_type = st.text_input("Enter the cuisine type (e.g., Japanese, American):")
    if st.button("Suggest Cuisine Pairing"):
        if cuisine_type:
            cuisine_pairing_chain = LLMChain(llm=llm_gen, prompt=prompt_cuisine_pairing)
            response = cuisine_pairing_chain.run({"cuisine_type": cuisine_type})
            st.write("### Suggested Cuisine Pairing:")
            st.write(response)
        else:
            st.write("Please enter a cuisine type.")


st.sidebar.info("Developed by Rachit Ranjan.")
