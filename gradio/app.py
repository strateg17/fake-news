import gradio as gr
import joblib
# from lightgbm import LGBMClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer (assuming they're saved as 'lgbm_model.pkl' and 'vectorizer.pkl')
model = joblib.load('lgbm_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

def classify_text(text):
    # Transform the input text using the loaded vectorizer
    text_vector = vectorizer.transform([text])
    
    # Predict using the loaded model
    prediction = model.predict(text_vector)
    
    return int(prediction[0])

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.Label(),
    title="Fake News Classifier",
    description="Enter text to classify if it's fake (1) or not fake (0).",
    examples=["This is a sample news article."]
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
