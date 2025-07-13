#!/usr/bin/env python3
"""
Dependency installation script for Medical QA System
Handles installation of required packages with proper error handling
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package with error handling."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def install_nltk_data():
    """Install required NLTK data."""
    try:
        print("Installing NLTK data...")
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✓ Successfully installed NLTK data")
        return True
    except Exception as e:
        print(f"✗ Failed to install NLTK data: {e}")
        return False

def main():
    """Main installation function."""
    print("Medical QA System - Dependency Installation")
    print("=" * 50)
    
    # Core packages that should work without compilation
    core_packages = [
        "fastapi==0.68.1",
        "uvicorn==0.15.0",
        "pandas==1.3.3",
        "numpy==1.21.2",
        "scikit-learn==0.24.2",
        "nltk==3.6.3",
        "matplotlib==3.4.3",
        "seaborn==0.11.2",
        "streamlit==0.84.2",
        "plotly==5.3.1",
        "faiss-cpu==1.7.2",
        "whoosh==2.7.4",
        "pydantic==1.8.2",
        "node2vec==0.3.3",
        "networkx==2.6.3",
        "shap==0.40.0",
        "requests==2.26.0",
        "python-multipart==0.0.5",
        "aiofiles==0.7.0",
        "python-jose==3.3.0",
        "passlib==1.7.4",
        "bcrypt==3.2.0",
        "python-dotenv==0.19.0",
        "textblob==0.17.1"
    ]
    
    # ML packages that might need special handling
    ml_packages = [
        "transformers==4.21.0",
        "torch==1.12.1"
    ]
    
    print("Installing core packages...")
    core_success = 0
    for package in core_packages:
        if install_package(package):
            core_success += 1
    
    print(f"\nCore packages: {core_success}/{len(core_packages)} installed successfully")
    
    print("\nInstalling ML packages...")
    ml_success = 0
    for package in ml_packages:
        if install_package(package):
            ml_success += 1
    
    print(f"ML packages: {ml_success}/{len(ml_packages)} installed successfully")
    
    # Install NLTK data
    print("\nInstalling NLTK data...")
    nltk_success = install_nltk_data()
    
    print("\n" + "=" * 50)
    print("Installation Summary:")
    print(f"Core packages: {core_success}/{len(core_packages)}")
    print(f"ML packages: {ml_success}/{len(ml_packages)}")
    print(f"NLTK data: {'✓' if nltk_success else '✗'}")
    
    if core_success >= len(core_packages) * 0.8:  # At least 80% success
        print("\n✓ Installation mostly successful! You can now run the system.")
        print("\nTo start the system:")
        print("1. Start the API: python app.py")
        print("2. Run evaluation: streamlit run evaluate.py")
    else:
        print("\n✗ Installation had issues. Please check the errors above.")
        print("You may need to install Microsoft Visual C++ Build Tools for some packages.")
        print("Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")

if __name__ == "__main__":
    main() 