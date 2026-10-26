import gradio as gr
from groq import Groq
import google.generativeai as genai

# Groq's Whisper requires ISO-639-1 language codes
LANG_CODES = {
    "Auto-Detect": None,
    "English": "en", "Spanish": "es", "French": "fr", 
    "German": "de", "Japanese": "ja", "Chinese": "zh", 
    "Korean": "ko", "Indonesian": "id"
}

def process_audio(audio_file, language, groq_key, gemini_key):
    if not groq_key or not gemini_key:
        return "⚠️ Please provide both Groq and Gemini API keys in the interface.", "No summary available."
    if not audio_file:
        return "⚠️ No audio provided. Please record or upload a file.", "No summary available."

    # Step A: Fast Transcription via Groq (Whisper-Large-v3)
    try:
        groq_client = Groq(api_key=groq_key)
        
        with open(audio_file, "rb") as f:
            # Build API arguments dynamically based on language choice
            kwargs = {"file": f, "model": "whisper-large-v3"}
            if LANG_CODES[language]:
                kwargs["language"] = LANG_CODES[language]
                
            transcription = groq_client.audio.transcriptions.create(**kwargs)
            transcript_text = transcription.text.strip()
            
        if not transcript_text:
            return "⚠️ No speech detected.", "No summary available."
            
    except Exception as e:
        return f"❌ Transcription Error: {str(e)}", "No summary available."

    # Step B: Fast Summarization via Google Gemini 1.5 Flash
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"You are a highly efficient assistant. Provide a clear, concise executive summary of this transcript:\n\n{transcript_text}"
        response = model.generate_content(prompt)
        summary_text = response.text.strip()
        
    except Exception as e:
        summary_text = f"❌ Summarization Error: {str(e)}"

    formatted_transcript = f"[Target: {language}]\n\n{transcript_text}"
    return formatted_transcript, summary_text

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️🎙️ Cloud-Powered Multilingual Transcriber")
    gr.Markdown("Runs via Groq (Whisper Large v3) and Gemini (1.5 Flash). Perfect for free CPU hosting.")
    
    with gr.Row():
        groq_input = gr.Textbox(label="Groq API Key", type="password", placeholder="Get free key at console.groq.com")
        gemini_input = gr.Textbox(label="Gemini API Key", type="password", placeholder="Get free key at aistudio.google.com")
        
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
            language_dropdown = gr.Dropdown(choices=list(LANG_CODES.keys()), value="Auto-Detect", label="Language (Optional)")
            submit_btn = gr.Button("🚀 Transcribe & Summarize", variant="primary")
            
        with gr.Column(scale=1):
            transcript_output = gr.Textbox(label="Transcription Result", lines=8)
            summary_output = gr.Textbox(label="AI Summary", lines=4)

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, language_dropdown, groq_input, gemini_input],
        outputs=[transcript_output, summary_output]
    )

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
