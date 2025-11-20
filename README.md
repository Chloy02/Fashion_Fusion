# 👗 FashionFusion

FashionFusion is an AI-powered personal stylist that transforms your closet into a digital runway. It uses advanced computer vision and large language models to classify your clothing and provide personalized, real-life style recommendations.

## ✨ Features

-   **Advanced Classification**: Uses **OpenAI's CLIP (Large)** model for zero-shot classification. It accurately identifies clothing items (e.g., "t-shirt", "dress", "sneakers") without needing custom training.
-   **AI Stylist**: Integrated with **Google Gemini 2.0 Flash** to generate dynamic, context-aware outfit suggestions based on:
    -   The specific item you uploaded.
    -   Occasion (Casual, Work, Party, etc.).
    -   Season (Spring, Summer, Fall, Winter).
    -   Style Preference (Masculine, Feminine, Neutral).
-   **Modern UI**: A sleek, responsive interface built with Streamlit.
-   **Secure**: API keys are managed securely using environment variables.

## 🚀 Getting Started

### Prerequisites

-   Python 3.11+
-   A Google Gemini API Key (Get one from [Google AI Studio](https://aistudio.google.com/))

### Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd Fashion_Fusion
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key**:
    -   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    -   Open `.env` and replace `your_api_key_here` with your actual Google Gemini API Key.
        ```env
        GOOGLE_API_KEY=AIzaSyD_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ```

### Usage

1.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser**: The app will typically run at `http://localhost:8501`.

3.  **Upload & Style**:
    -   Upload an image of a clothing item.
    -   Select your preferences (Occasion, Season, Style) in the sidebar.
    -   Click **"✨ Generate Style Suggestions"**.

## 🧠 AI Models Used

-   **Classification**: `openai/clip-vit-large-patch14`
    -   *Why?* It understands images and text together, allowing it to identify fashion items with high accuracy without specific training on a fashion dataset.
-   **Suggestions**: `gemini-2.0-flash`
    -   *Why?* It provides fast, creative, and high-quality text generation, acting as an intelligent fashion consultant.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **ML/AI**: PyTorch, Transformers (Hugging Face), Google Generative AI SDK
-   **Image Processing**: Pillow (PIL)