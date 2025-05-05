import streamlit as st
import os
import sys
from app_ui import setup_ui
from model_loader import ModelLoader

def main():
    """
    Main function for the Smart Traffic Light Control System.
    Sets up the UI and handles the main application flow.
    """
    # Initialize session state if not already done
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
    
    if 'using_example' not in st.session_state:
        st.session_state.using_example = False
    
    # Setup the UI
    setup_ui()

if __name__ == "__main__":
    main()
