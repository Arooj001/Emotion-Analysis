import gradio as gr
from transformers import pipeline

# Load the model for emotion detection
classifier = pipeline(
    "text-classification",
    model='bhadresh-savani/distilbert-base-uncased-emotion',
    return_all_scores=True
)

def detect_emotions(emotion_input):
    """
    Detect emotions in the input text using a pre-trained model.
    Returns a dictionary mapping emotions to their respective scores.
    """
    prediction = classifier(emotion_input)
    output = {emotion["label"]: round(emotion["score"], 4) for emotion in prediction[0]}
    return output

examples = [
    ["Every song on the radio reminds me of you."],
    ["There's an unfamiliar shadow in the corner of the room."]
]

css = """
footer {display: none !important;}
.output-markdown {display: none !important;}
.gr-button-primary {
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important;
    background: rgb(17, 20, 45) !important;
    border: none !important;
    text-align: center !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: none !important;
}
.gr-button-primary:hover {
    background: rgb(66, 133, 244) !important;
    box-shadow: rgb(0 0 0 / 23%) 0px 1px 7px 0px !important;
}
"""

interface = gr.Interface(
    fn=detect_emotions,
    inputs=gr.Textbox(placeholder="Enter text here", label="Input", lines=2),
    outputs=gr.Label(num_top_classes=5, label="Emotion"),
    title="Emotion Analysis",
    description="Enter a text to detect the underlying emotions using a DistilBERT-based model.",
    examples=examples,
    css=css
)

if __name__ == "__main__":
    interface.launch()
