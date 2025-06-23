import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import re

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and clean dataset
df = pd.read_csv(r"C:\6th Semester\NLP\amazon.csv")
df = df.drop_duplicates()
df["combined_text"] = df["product_name"].fillna("") + " " + df["about_product"].fillna("")
df["clean_price"] = df["discounted_price"].replace('[₹,]', '', regex=True).astype(float)

# Precompute embeddings
df["embedding"] = df["combined_text"].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Price parser from text
def extract_price(text):
    text = text.lower()
    match = re.search(r"(under|below|less than|over|above|more than)\s?₹?([\d,]+)", text)
    if match:
        keyword, num = match.groups()
        num = int(num.replace(",", ""))
        direction = "under" if keyword in ["under", "below", "less than"] else "over"
        return direction, num
    return None, None

# Main response function
def chatbot(user_input, chat_history):
    direction, limit = extract_price(user_input)

    filtered_df = df.copy()
    if direction and limit:
        if direction == "under":
            filtered_df = filtered_df[filtered_df["clean_price"] <= limit]
        else:
            filtered_df = filtered_df[filtered_df["clean_price"] >= limit]

    if filtered_df.empty:
        reply = "No products found with that price range."
        chat_history.append((user_input, reply))
        return "", chat_history, 0

    query_embedding = model.encode(user_input, convert_to_tensor=True)
    filtered_df["similarity"] = filtered_df["embedding"].apply(lambda emb: util.pytorch_cos_sim(query_embedding, emb).item())
    top_matches = filtered_df.sort_values(by="similarity", ascending=False).head(3)

    reply = "**Top Product Suggestions:**\n\n"
    for i, (_, row) in enumerate(top_matches.iterrows(), 1):
        reply += (
            f"**{i}. {row['product_name']}**\n"
            f"- 💸 Price: ₹{row['clean_price']:.0f} (was {row['actual_price']}, {row['discount_percentage']} off)\n"
            f"- ⭐ Rating: {row['rating']} ({row['rating_count']} reviews)\n"
            f"- [🔗 View Product]({row['product_link']})\n\n"
        )

    chat_history.append((user_input, reply))
    return "", chat_history, top_matches.index[0]  # First of top 3 is the default for "order"

# Cart handling
cart = []

def handle_order(message, chat_history, last_suggested_index):
    if message.strip().lower() == "order":
        product = df.loc[last_suggested_index]
        cart.append(product["product_name"])
        chat_history.append(("order", f"✅ **{product['product_name']}** has been added to your cart."))
        return "", chat_history, last_suggested_index
    else:
        return chatbot(message, chat_history)

# Wrapper
def respond(message, chat_history=[], last_suggested_index=0):
    return handle_order(message, chat_history, last_suggested_index)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🛍️ Product Chatbot\nAsk for suggestions like 'charger under 500' or 'cable above 300'")
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Type your message")
    state_idx = gr.State(0)
    state_chat = gr.State([])

    msg.submit(respond, [msg, state_chat, state_idx], [msg, chatbot_ui, state_idx])

demo.launch()
