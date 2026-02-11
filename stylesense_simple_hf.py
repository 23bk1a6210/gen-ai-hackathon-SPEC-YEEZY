"""
StyleSense: AI Fashion Recommendation System (Hugging Face - FREE)
"""

import gradio as gr
from huggingface_hub import InferenceClient

def get_advice(query, style, occasion, colors, budget, api_key):
    if not api_key:
        return "⚠️ Enter your Hugging Face API key!"
    if not query.strip():
        return "⚠️ Ask a question!"
    
    try:
        client = InferenceClient(token=api_key)
        
        prompt = f"""You are a fashion stylist. User asks: "{query}"

Style preference: {style}
Occasion: {occasion}
Preferred colors: {colors}
Budget: {budget}

Provide specific fashion advice with outfit suggestions."""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Llama-3.2-3B-Instruct",
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="StyleSense") as app:
    gr.Markdown("# 👔 StyleSense AI Fashion Advisor")
    gr.Markdown("*FREE - Powered by Hugging Face Mistral-7B*")
    
    api_key = gr.Textbox(
        label="🔑 Hugging Face API Key", 
        type="password", 
        placeholder="Get free at huggingface.co/settings/tokens"
    )
    
    query = gr.Textbox(
        label="Ask Your Fashion Question",
        lines=3, 
        placeholder="e.g., What should I wear to a summer wedding? How do I style a leather jacket?"
    )
    
    with gr.Row():
        style = gr.Dropdown(
            ["Casual", "Formal", "Business Casual", "Streetwear", "Minimalist", "Bohemian"], 
            label="Style", 
            value="Casual"
        )
        occasion = gr.Dropdown(
            ["Daily Wear", "Office", "Party", "Wedding", "Date Night", "Workout"], 
            label="Occasion", 
            value="Daily Wear"
        )
    
    with gr.Row():
        colors = gr.Textbox(label="Preferred Colors", value="Any", placeholder="e.g., Navy, Beige, Pastels")
        budget = gr.Dropdown(
            ["Budget-Friendly", "Medium", "Premium", "Luxury"], 
            label="Budget", 
            value="Medium"
        )
    
    ask_btn = gr.Button("✨ Get Fashion Advice", variant="primary", size="lg")
    output = gr.Markdown(label="Recommendations")
    
    ask_btn.click(
        get_advice, 
        inputs=[query, style, occasion, colors, budget, api_key], 
        outputs=output
    )
    
    gr.Markdown("""
    ### Tips:
    - Ask specific questions for best results
    - Describe what you already have in your wardrobe
    - Mention body type or preferences for personalized advice
    """)

app.launch(share=True)