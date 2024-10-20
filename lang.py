from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
import streamlit_mermaid as stmd

# Initialize the Llama model using Ollama
llm = Ollama(model="llama3.2")

def strip_mermaid_code(input_code):
    stripped_code = input_code.strip()
    stripped_code = stripped_code.strip('`')
    lines = stripped_code.split('\n')
    if lines and lines[0].lower().startswith('mermaid'):
        lines.pop(0)
    return '\n'.join(lines)

# Create a prompt template
prompt_template = PromptTemplate(
    template="""Generate a Mermaid code based on the following description. 
    Ensure the code is compatible with Mermaid version 10.2.4 and free of syntax errors. 
    Do not give any explanantion, provide only the code.

    Description: {text}
    """,
    input_variables=["text"]
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt_template)


# Streamlit UI
st.title("Mermaid Diagram Generator")
text = st.text_input('Enter your diagram description:')

if text:
    result = chain.run(text=text)
    try:
        output_json = {
            "code": "",
            "explanation": ""
        }
        st.subheader("Raw Output:")
        code = strip_mermaid_code(result)
        # st.code(code)
        
        

        explanation_template = PromptTemplate(
            template="""Based on the following code, only explain the code in few steps, do not add any other sentences that may be fillers.

            Code: {result}
            """,
            input_variables=["result"]
        )
        chain2 = LLMChain(llm=llm, prompt=explanation_template)
        result2 = chain2.run(result=result)
        output_json["code"] = code
        output_json["explanation"] = result2
        st.write(output_json)
        # st.write(result2)

        
    except Exception as e:
        st.error(f"Error processing output: {e}")
