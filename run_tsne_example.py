#!/usr/bin/env python3
"""
Example script demonstrating t-SNE analysis for ICD-9 embeddings
===============================================================

This script shows how to run the t-SNE analysis with different parameters
and configurations for evaluating the semantic quality of ICD-9 embeddings.

Usage:
    python run_tsne_example.py
"""

import subprocess
import sys
import os

def run_analysis_with_params(perplexity, iterations, output_dir):
    """Run t-SNE analysis with specified parameters."""
    print(f"\n🔬 Running t-SNE analysis with perplexity={perplexity}, iterations={iterations}")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis
    cmd = [
        sys.executable, "quick_tsne_analysis.py",
        "--perplexity", str(perplexity),
        "--iterations", str(iterations),
        "--output-dir", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Analysis completed successfully!")
        print(f"📊 Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function demonstrating different analysis configurations."""
    print("🧬 ICD-9 Embedding Semantic Analysis Examples")
    print("=" * 50)
    
    # Example 1: Default analysis
    print("\n📋 Example 1: Default Analysis")
    print("This uses the recommended parameters for most datasets.")
    success1 = run_analysis_with_params(
        perplexity=30,
        iterations=1000,
        output_dir="./tsne_results_default"
    )
    
    # Example 2: High detail analysis
    print("\n📋 Example 2: High Detail Analysis")
    print("This uses higher perplexity and more iterations for better detail.")
    success2 = run_analysis_with_params(
        perplexity=50,
        iterations=2000,
        output_dir="./tsne_results_high_detail"
    )
    
    # Example 3: Quick analysis
    print("\n📋 Example 3: Quick Analysis")
    print("This uses lower parameters for faster processing.")
    success3 = run_analysis_with_params(
        perplexity=20,
        iterations=500,
        output_dir="./tsne_results_quick"
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    results = [
        ("Default Analysis", success1, "./tsne_results_default"),
        ("High Detail Analysis", success2, "./tsne_results_high_detail"),
        ("Quick Analysis", success3, "./tsne_results_quick")
    ]
    
    for name, success, output_dir in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if success and os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"  📁 Files: {', '.join(files)}")
    
    print("\n🎯 Next Steps:")
    print("1. Review the generated visualizations")
    print("2. Compare results between different parameter settings")
    print("3. Use the interactive Streamlit app for detailed exploration:")
    print("   streamlit run tsne_embedding_analysis.py")
    print("4. Export results for publication or further analysis")

if __name__ == "__main__":
    main() 