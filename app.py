"""
House Price Prediction - Gradio Web Application
Loads the trained model pipeline and provides an interactive web interface
for predicting house prices based on area features.
"""

import gradio as gr
import joblib
import pandas as pd
import os

# Load the trained pipeline
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
pipeline = joblib.load(model_path)

# Feature information for the UI
feature_info = {
    'Avg. Area Income': {'min': 17797, 'max': 107702, 'default': 68583, 'step': 100,
                          'label': 'Avg. Area Income ($)', 'desc': 'Average income of residents in the area'},
    'Avg. Area House Age': {'min': 2.6, 'max': 9.5, 'default': 6.0, 'step': 0.1,
                             'label': 'Avg. Area House Age (years)', 'desc': 'Average age of houses in the area'},
    'Avg. Area Number of Rooms': {'min': 3.0, 'max': 11.0, 'default': 6.0, 'step': 0.1,
                                   'label': 'Avg. Area Number of Rooms', 'desc': 'Average number of rooms in the area'},
    'Avg. Area Number of Bedrooms': {'min': 2.0, 'max': 7.0, 'default': 3.0, 'step': 0.1,
                                      'label': 'Avg. Area Number of Bedrooms', 'desc': 'Average number of bedrooms in the area'},
    'Area Population': {'min': 173, 'max': 69622, 'default': 36164, 'step': 100,
                         'label': 'Area Population', 'desc': 'Population of the area'}
}


def predict_price(income, house_age, num_rooms, num_bedrooms, population):
    """
    Predict house price based on input features.
    """
    input_data = pd.DataFrame({
        'Avg. Area Income': [income],
        'Avg. Area House Age': [house_age],
        'Avg. Area Number of Rooms': [num_rooms],
        'Avg. Area Number of Bedrooms': [num_bedrooms],
        'Area Population': [population]
    })
    prediction = pipeline.predict(input_data)[0]
    return f"Predicted House Price: ${prediction:,.2f}"


# Example inputs for the Gradio interface
examples = [
    [79545, 5.9, 6.5, 3.2, 36164],
    [61287, 6.0, 7.5, 4.0, 42000],
    [68583, 5.9, 6.0, 3.0, 36164],
    [55000, 4.5, 5.0, 2.5, 25000],
]

# Create the Gradio interface
with gr.Blocks(title="House Price Prediction") as demo:

    gr.HTML("<h1 style='text-align: center; margin-bottom: 5px;'>House Price Prediction</h1>")
    gr.HTML("<p style='text-align: center; color: #6B7280; margin-bottom: 20px;'>Predict house prices based on area features using machine learning</p>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input Features")
            gr.Markdown("Enter the area characteristics to predict the house price.")

            income = gr.Number(
                label=feature_info['Avg. Area Income']['label'],
                value=feature_info['Avg. Area Income']['default'],
                info=feature_info['Avg. Area Income']['desc']
            )

            house_age = gr.Slider(
                minimum=feature_info['Avg. Area House Age']['min'],
                maximum=feature_info['Avg. Area House Age']['max'],
                value=feature_info['Avg. Area House Age']['default'],
                step=feature_info['Avg. Area House Age']['step'],
                label=feature_info['Avg. Area House Age']['label'],
                info=feature_info['Avg. Area House Age']['desc']
            )

            num_rooms = gr.Slider(
                minimum=feature_info['Avg. Area Number of Rooms']['min'],
                maximum=feature_info['Avg. Area Number of Rooms']['max'],
                value=feature_info['Avg. Area Number of Rooms']['default'],
                step=feature_info['Avg. Area Number of Rooms']['step'],
                label=feature_info['Avg. Area Number of Rooms']['label'],
                info=feature_info['Avg. Area Number of Rooms']['desc']
            )

            num_bedrooms = gr.Slider(
                minimum=feature_info['Avg. Area Number of Bedrooms']['min'],
                maximum=feature_info['Avg. Area Number of Bedrooms']['max'],
                value=feature_info['Avg. Area Number of Bedrooms']['default'],
                step=feature_info['Avg. Area Number of Bedrooms']['step'],
                label=feature_info['Avg. Area Number of Bedrooms']['label'],
                info=feature_info['Avg. Area Number of Bedrooms']['desc']
            )

            population = gr.Number(
                label=feature_info['Area Population']['label'],
                value=feature_info['Area Population']['default'],
                info=feature_info['Area Population']['desc']
            )

        with gr.Column(scale=1):
            gr.Markdown("### Prediction Result")
            output = gr.Textbox(label="")

            predict_btn = gr.Button("Predict Price", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", variant="secondary")

            gr.Markdown("""
            ### How It Works
            This app uses a machine learning model trained on the USA Housing dataset.
            The model analyzes area-level features like income, house age, room counts,
            and population to predict house prices.

            **Tips:**
            - Higher area income generally leads to higher house prices
            - Newer houses (lower age) tend to be more expensive
            - More rooms typically indicate higher prices
            """)

    gr.Markdown("### Example Inputs")
    gr.Examples(
        examples=examples,
        inputs=[income, house_age, num_rooms, num_bedrooms, population],
        outputs=output,
        fn=predict_price,
    )

    predict_btn.click(
        fn=predict_price,
        inputs=[income, house_age, num_rooms, num_bedrooms, population],
        outputs=output
    )

    clear_btn.click(
        fn=lambda: (feature_info['Avg. Area Income']['default'],
                     feature_info['Avg. Area House Age']['default'],
                     feature_info['Avg. Area Number of Rooms']['default'],
                     feature_info['Avg. Area Number of Bedrooms']['default'],
                     feature_info['Area Population']['default'],
                     ""),
        inputs=None,
        outputs=[income, house_age, num_rooms, num_bedrooms, population, output]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft()
    )