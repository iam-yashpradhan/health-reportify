import streamlit as st
import streamlit_mermaid as stmd
import ollama



def strip_mermaid_code(input_code):
    # Strip leading and trailing whitespace
    stripped_code = input_code.strip()
    
    # Remove starting and ending triple backticks
    stripped_code = stripped_code.strip('`')
    
    # Split the code into lines
    lines = stripped_code.split('\n')
    
    # Remove the first line if it defines the language
    if lines and lines[0].lower().startswith('mermaid'):
        lines.pop(0)
    
    # Join the remaining lines back into a single string
    return '\n'.join(lines)

# if text is not None:
#     response = ollama.chat(
#                 model='llama3.2:3b',
#                 messages=[{
#     'role': 'user',
#     'content': f'''Please generate Mermaid diagram code based on the following description. Ensure the code is compatible with Mermaid version 10.2.4 and free of syntax errors. Additionally, provide a brief explanation of the Mermaid code. The output should be in the format given below which says Output Format with two attributes: "mermaid_code" and "explanation".
# Description: {text}
# Output Format:
# {{
#   "mermaid_code": "Your Mermaid code here",
#   "explanation": "A brief explanation of the Mermaid code here"
# }} '''}])['message']['content']
#     st.write(response)
#     # code = strip_mermaid_code(response)
#     # print(code)
#     # stmd.st_mermaid(code)

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain

# Initialize the Llama model using Ollama
llm = Ollama(model="llama3.2")

# Create a JSON output parser
json_parser = JsonOutputParser()

# Create a prompt template
prompt_template = PromptTemplate(
    template="Answer the following question. {format_instructions}\n\nQuestion: {question}",
    input_variables=["question"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)
text = st.text_input('Text here')
# Function to process the prompt and return JSON
def process_prompt(question):
    result = chain.run(question=question)
    return json_parser.parse(result)

if text is not None:
    json_output = process_prompt(text)
    print(json_output)

