"""
Project Sentinel V2 - UI Styles
-------------------------------
Centralized CSS styles for the Streamlit dashboard.
"""

MAIN_STYLES = """
<style>
    :root {
        --bg-color: #0e1117;
        --card-bg: rgba(30, 35, 45, 0.7);
        --card-border: rgba(255, 255, 255, 0.1);
        --primary: #3b82f6;
        --accent: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
    }

    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
    }

    /* FIX: Visibility for Select Box */
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    .stMultiSelect div[data-baseweb="tag"] {
        background-color: var(--primary) !important;
    }

    /* Custom Cards */
    .sentinel-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease-in-out;
        margin-bottom: 1rem;
    }
    
    .sentinel-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .lifeline-card {
        border-left: 4px solid var(--danger);
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(30, 35, 45, 0.7) 100%);
    }

    .card-header {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .card-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
    }
    
    /* Image Styling */
    img {
        border-radius: 8px;
        transition: transform 0.2s;
    }
    
    img:hover {
        transform: scale(1.02);
    }

    /* Branding Hiding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
