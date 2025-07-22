#!/usr/bin/env python3
"""
Example usage script for the adaptive grid visualizer.
This script demonstrates how to generate visualization images for the adaptive grid algorithm.
"""

from adaptive_grid_visualizer import visualize_adaptive_grid_steps
import json
import os

def run_visualization_examples():
    """Run several examples with different configurations"""
    
    # Example 1: Basic usage with a chart image (if available)
    # Replace this with any image path you want to test
    example_images = [
        "/Users/minsukchang/Research/ChartDataset/chartqa/05411753006467.png",
        "/Users/minsukchang/Research/ChartDataset/chartqa/9280.png",
        "/Users/minsukchang/Research/ChartDataset/charxiv/17.jpg"
    ]
    
    # Load OCR dictionary if available
    ocr_dict = {}
    ocr_files = [
        "/Users/minsukchang/Research/ChartDataset/OCR/chartqa.json",
        "/Users/minsukchang/Research/ChartDataset/OCR/charxiv.json"
    ]
    
    for ocr_file in ocr_files:
        try:
            with open(ocr_file) as f:
                ocr_dict.update(json.load(f))
                print(f"Loaded OCR data from {ocr_file}")
        except FileNotFoundError:
            print(f"OCR file {ocr_file} not found, skipping")
    
    for i, image_path in enumerate(example_images):
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found, skipping")
            continue
            
        print(f"\n--- Processing Example {i+1}: {os.path.basename(image_path)} ---")
        
        # Configuration for this example
        config = {
            "output_folder": f"example_{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}",
            "min_grid_ratio": 0.08,
            "tile_shape": "rect",
            "max_tile": 10,
            "text_color": (255, 100, 100),    # Red for text areas
            "edge_color": (100, 255, 100),    # Green for edge areas  
            "blank_color": (100, 100, 255),   # Blue for blank areas
            "grid_color": (64, 64, 64),       # Dark gray for final grid
            "ocr_dict": ocr_dict
        }
        
        try:
            rectangles, stats = visualize_adaptive_grid_steps(
                image_path=image_path,
                **config
            )
            print(f"✅ Successfully generated {stats['total_tiles']} tiles")
            print(f"   - Background: {stats['background_tiles']}")
            print(f"   - Edge: {stats['edge_tiles']}")
            print(f"   - Text: {stats['text_tiles']}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")

def custom_example():
    """Example with custom parameters and any image"""
    
    # Replace with your image path
    image_path = input("Enter the path to your image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found!")
        return
    
    print("\n--- Custom Configuration Example ---")
    
    # Custom configuration
    rectangles, stats = visualize_adaptive_grid_steps(
        image_path=image_path,
        output_folder="custom_visualization",
        min_grid_ratio=0.1,               # Larger base grid
        tile_shape="rect",                # or "square"
        max_tile=8,                       # Smaller max tile size
        text_color=(200, 50, 50),         # Dark red for text
        edge_color=(50, 200, 50),         # Dark green for edges
        blank_color=(50, 50, 200),        # Dark blue for blank
        grid_color=(100, 100, 100),       # Gray for final grid
        ocr_dict={}                       # No OCR data
    )
    
    print(f"✅ Custom visualization complete!")
    print(f"Generated {stats['total_tiles']} tiles total")

def batch_processing_example():
    """Example of processing multiple images with same configuration"""
    
    # Add your image directory path here
    image_dir = "/Users/minsukchang/Research/ChartDataset/chartqa"
    
    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} not found!")
        return
    
    # Get all PNG and JPG files
    import glob
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + \
                  glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.jpeg"))
    
    # Limit to first 3 images for demo
    image_files = image_files[:3]
    
    print(f"\n--- Batch Processing {len(image_files)} Images ---")
    
    # Load OCR data
    ocr_dict = {}
    try:
        with open("/Users/minsukchang/Research/ChartDataset/OCR/chartqa.json") as f:
            ocr_dict = json.load(f)
    except FileNotFoundError:
        print("No OCR data found")
    
    for image_path in image_files:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessing: {basename}")
        
        try:
            rectangles, stats = visualize_adaptive_grid_steps(
                image_path=image_path,
                output_folder=f"batch_output/{basename}",
                min_grid_ratio=0.08,
                tile_shape="rect",
                max_tile=10,
                ocr_dict=ocr_dict
            )
            print(f"  ✅ Generated {stats['total_tiles']} tiles")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("Adaptive Grid Visualization Examples")
    print("=" * 40)
    
    print("\n1. Running predefined examples...")
    run_visualization_examples()
    
    print("\n2. Want to try a custom image? (y/n)")
    if input().lower().startswith('y'):
        custom_example()
    
    print("\n3. Want to run batch processing? (y/n)")
    if input().lower().startswith('y'):
        batch_processing_example()
    
    print("\n🎉 All done! Check the output folders for visualization images.") 