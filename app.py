import streamlit as st
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates
import pandas as pd
import json
from PIL import Image
import base64
import io

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model="llama3.2")

def generate_color():
    """Generates a random color in hex format."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def assign_colors_to_values(values_list):
    """Assigns a unique color to each unique individual value."""
    color_map = {}
    for value in values_list:
        color_map[value] = generate_color()
    return color_map

# Extract unique medicines and assign colors
def map_values_to_colors(values_list, color_map):
    """Maps each value in the list to its corresponding color."""
    color_output = []
    for val in values_list:
        color_output.append(color_map[val.strip()])
    return color_output

# Title for the Streamlit app
st.title('Medical Report Visualization and Chatbot')

# Create two tabs: one for the plot and one for the chatbot
tab1, tab2, tab3 = st.tabs(["Plot", "Recommendation", "X-Ray"])

# File uploader for both JSON file and X-ray image
uploaded_json = st.sidebar.file_uploader("Upload your JSON file", type="json")
uploaded_image = st.sidebar.file_uploader("Upload your X-ray image", type=["png", "jpg", "jpeg"])


if uploaded_json is not None:
    # Process the JSON file
    df = pd.read_json(uploaded_json)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df = df.head(30)
    # st.dataframe(df)
    report_data = df.copy()
    

    # Extract unique medicines
    unique_medicines = pd.Series([med.strip() for sublist in df['medicines_taken'] for med in sublist.split(',')]).unique()

    # Convert date column to datetime format
    df['date_num'] = df['date'].apply(lambda date: mdates.date2num(date))

    with tab1:
        st.header("Blood Report Visualization with Medicine Highlighting (Lines and Dots)")

        # Dropdown to select which attribute to plot
        columns = ['RBC', 'platelets', 'hemoglobin']
        attribute = st.selectbox('Select the attribute to plot', columns, index=0)

        # Get the selected attribute values for y-axis
        y = df[attribute]

        # Generate the plot using matplotlib
        fig, ax = plt.subplots()

        # Plot the y as a line plot
        x = df['date_num']
        ax.plot(x, y, marker='o', markersize=0, label="Line Plot with Points")  # Line plot with markers

        # Assign colors to unique medicines
        color_map = assign_colors_to_values(unique_medicines)

        # Define size dictionary for the pie charts
        dic_size = {'RBC': 2, 'hemoglobin': 2, 'platelets': 20000}

        # Loop through each unique medicine
        for i in range(len(x)):
            size = dic_size[attribute]  # Size of the pie chart
            inset_ax = ax.inset_axes([x[i] - size / 2, y[i] - size / 2, size, size], transform=ax.transData)
            temp_arr = df['medicines_taken'][i].split(',')

            # Create the pie chart in the inset axis with correct colors
            colors = map_values_to_colors(temp_arr, color_map)
            inset_ax.pie([1] * len(temp_arr), colors=colors)
            inset_ax.set_aspect('equal')

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically find the best position for date ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlabel('Date')
        ax.set_ylabel(attribute)
        plt.xticks(rotation=45, ha='right')

        # Medicine color guide
        html_content = "<div><strong>Medicine Color Guide:</strong><br>"
        for med, color in color_map.items():
            html_content += f"<div style='display: flex; align-items: center; margin: 5px;'><div style='width: 20px; height: 20px; background-color: {color}; margin-right: 5px; border: 1px solid black;'></div>{med}</div>"
        html_content += "</div>"

        # Display the HTML in the Streamlit app
        st.markdown(html_content, unsafe_allow_html=True)

        # Display the plot
        st.pyplot(fig)

       
        

    with tab2:
        st.header("Recommendation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label = "Hemoglobin", value = "12-18 g/dL" )
        with col2:
            st.metric(label = "Platelets", value = "0.15-0.45m" )
        with col3:
            st.metric(label = "RBCs", value = "3.5 - 5.5m" )

        # Simple chatbot implementation using Streamlit
        # user_input = st.text_input("Ask a question:")
        prompt_template = PromptTemplate(
            template="""Analyze the historic data below and list the medicines that should be taken and those that should not be taken based on the counts of RBCs, platelets, and hemoglobin.
            Display only the Output format and nothing else in your output

    Data: {report_data}

    Output format:
    - Medicines to take: [list of medicines]
    - Medicines not to take: [list of medicines]
    """,
    input_variables=["report_data"]
)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        result = chain.run(report_data=report_data)
        st.write(result)
    
    with tab3:
        st.header("X-Ray Diagnosis")
        run_diagnosis = st.button("Run Diagnosis")
        if run_diagnosis:
            if uploaded_image is not None:
            # Display the uploaded image
                # st.image(uploaded_image, caption="Uploaded X-ray Image", use_column_width=True)
                import skimage
                import torch
                import torch.nn.functional as F
                import torchvision
                import torchvision.transforms
                import torchxrayvision as xrv

                model_name = "densenet121-res224-mimic_nb"
                img_url = uploaded_image
                model = xrv.models.get_model(model_name)

                img = skimage.io.imread(img_url)
                img = xrv.datasets.normalize(img, 255)

                # Check that images are 2D arrays
                if len(img.shape) > 2:
                    img = img[:, :, 0]
                if len(img.shape) < 2:
                    print("error, dimension lower than 2 for image")

                # Add color channel
                img = img[None, :, :]

                transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

                img = transform(img)

                with torch.no_grad():
                    img = torch.from_numpy(img).unsqueeze(0)
                    preds = model(img).cpu()
                    output = {
                        k: float(v)
                        for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
                    }
                # st.write(output)
                identified_diseases = [
                    "Atelectasis",
                    "Pneumothorax",
                    "Edema",
                    "Emphysema",
                    "Fibrosis",
                    "Pneumonia"
                ]
                separated_diseases = {key: value for key, value in output.items() if key in identified_diseases}
                lung_health = {key: value for key, value in output.items() if key not in identified_diseases}
                top_2_diseases = sorted(separated_diseases.items(), key=lambda x: x[1], reverse=True)[:2]
                top_5_symptoms = sorted(lung_health.items(), key=lambda x: x[1], reverse=True)[:5]
                st.write(top_2_diseases[0][0], round(top_2_diseases[0][1], 3))
                # st.write(top_5_symptoms)
                symptoms_df = pd.DataFrame(top_5_symptoms, columns = ['Condition', 'Probability'])
                st.subheader("Top Signs and Symptoms")
                st.dataframe(symptoms_df)



else:
    st.write("Please upload a JSON file to see the plot.")
