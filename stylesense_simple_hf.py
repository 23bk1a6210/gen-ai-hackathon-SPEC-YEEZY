"""
StyleSense: AI Fashion Recommendation System with Image Generation
"""

import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image
import io

def generate_outfit_images(uploaded_image, style, occasion, colors, budget, api_key):
    if not api_key:
        return None, None, None, "⚠️ Enter your Hugging Face API key!"
    if uploaded_image is None:
        return None, None, None, "⚠️ Upload an image!"
    
    try:
        client = InferenceClient(token=api_key)
        
        # Generate 3 outfit suggestions as images
        prompts = [
            f"Fashion photography, {style} outfit for {occasion}, featuring {colors} colors, {budget} aesthetic, professional product photo, clean background, high quality",
            f"Stylish {style} ensemble for {occasion}, {colors} color palette, {budget} fashion, catalog style photo, well-lit, detailed",
            f"Modern {style} look perfect for {occasion}, incorporating {colors} tones, {budget} range, fashion magazine quality, professional styling"
        ]
        
        images = []
        descriptions = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Generate image using Stable Diffusion
                img_bytes = client.text_to_image(
                    prompt,
                    model="stabilityai/stable-diffusion-2-1"
                )
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)
                descriptions.append(f"**Outfit {i+1}:** {style} style for {occasion}")
            except:
                images.append(None)
                descriptions.append(f"**Outfit {i+1}:** Generation failed")
        
        # Generate text recommendations
        text_prompt = f"""As a fashion stylist, provide recommendations for:
Style: {style}
Occasion: {occasion}
Colors: {colors}
Budget: {budget}

Give 3 specific outfit ideas with:
- Key pieces to wear
- How to style them
- Accessories suggestions"""

        response = client.chat_completion(
            messages=[{"role": "user", "content": text_prompt}],
            model="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        
        recommendations = response.choices[0].message.content
        
        return images[0], images[1], images[2], recommendations
        
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"

def get_text_advice(query, style, occasion, colors, budget, api_key):
    if not api_key:
        return "⚠️ Enter your Hugging Face API key!"
    if not query.strip():
        return "⚠️ Ask a question!"
    
    try:
        client = InferenceClient(token=api_key)
        
        prompt = f"""You are a fashion stylist. User asks: "{query}"

Preferences: Style={style}, Occasion={occasion}, Colors={colors}, Budget={budget}

Provide specific fashion advice with outfit suggestions."""

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(title="StyleSense", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 👔 StyleSense: AI Fashion Generator")
    gr.Markdown("*Upload your photo, choose preferences, get AI-generated outfit suggestions!*")
    
    api_key = gr.Textbox(
        label="🔑 Hugging Face API Key", 
        type="password", 
        placeholder="Get free at huggingface.co/settings/tokens"
    )
    
    with gr.Tabs():
        # Image Generation Tab
        with gr.Tab("✨ Generate Outfits"):
            gr.Markdown("### Upload your reference image and get 3 AI-generated outfit suggestions!")
            
            with gr.Row():
                with gr.Column(scale=1):
                    uploaded_img = gr.Image(type="pil", label="Your Reference Image")
                    
                    style_gen = gr.Dropdown(
                        ["Casual", "Formal", "Business Casual", "Streetwear", "Minimalist", "Bohemian", "Sporty"], 
                        label="Style", 
                        value="Casual"
                    )
                    occasion_gen = gr.Dropdown(
                        ["Daily Wear", "Office", "Party", "Wedding", "Date Night", "Workout", "Travel"], 
                        label="Occasion", 
                        value="Daily Wear"
                    )
                    colors_gen = gr.Textbox(
                        label="Preferred Colors", 
                        value="neutral tones",
                        placeholder="e.g., pastels, earth tones, monochrome"
                    )
                    budget_gen = gr.Dropdown(
                        ["Budget-Friendly", "Medium", "Premium", "Luxury"], 
                        label="Budget", 
                        value="Medium"
                    )
                    
                    generate_btn = gr.Button("🎨 Generate 3 Outfit Ideas", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### AI-Generated Outfit Suggestions")
                    
                    with gr.Row():
                        outfit1 = gr.Image(label="Outfit 1", height=300)
                        outfit2 = gr.Image(label="Outfit 2", height=300)
                        outfit3 = gr.Image(label="Outfit 3", height=300)
                    
                    recommendations = gr.Markdown(label="Detailed Recommendations")
            
            generate_btn.click(
                generate_outfit_images,
                inputs=[uploaded_img, style_gen, occasion_gen, colors_gen, budget_gen, api_key],
                outputs=[outfit1, outfit2, outfit3, recommendations]
            )
        
        # Text Chat Tab
        with gr.Tab("💬 Ask Stylist"):
            gr.Markdown("### Ask any fashion question for personalized advice")
            
            query = gr.Textbox(
                label="Your Fashion Question",
                lines=3, 
                placeholder="e.g., What should I wear to a summer wedding?"
            )
            
            with gr.Row():
                style_text = gr.Dropdown(
                    ["Casual", "Formal", "Business Casual", "Streetwear", "Minimalist", "Bohemian"], 
                    label="Style", 
                    value="Casual"
                )
                occasion_text = gr.Dropdown(
                    ["Daily Wear", "Office", "Party", "Wedding", "Date Night"], 
                    label="Occasion", 
                    value="Daily Wear"
                )
            
            with gr.Row():
                colors_text = gr.Textbox(label="Preferred Colors", value="Any")
                budget_text = gr.Dropdown(
                    ["Budget-Friendly", "Medium", "Premium", "Luxury"], 
                    label="Budget", 
                    value="Medium"
                )
            
            ask_btn = gr.Button("✨ Get Fashion Advice", variant="primary", size="lg")
            text_output = gr.Markdown(label="AI Recommendations")
            
            ask_btn.click(
                get_text_advice, 
                inputs=[query, style_text, occasion_text, colors_text, budget_text, api_key], 
                outputs=text_output
            )
    
    gr.Markdown("""
    ---
    ### How it works:
    1. **Upload** a reference image (outfit inspiration, body type, style reference)
    2. **Choose** your preferences (style, occasion, colors, budget)
    3. **Generate** 3 AI-created outfit images matching your preferences
    4. Get detailed styling recommendations!
    """)

app.launch(share=True)
