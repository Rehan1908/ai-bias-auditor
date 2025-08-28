import os
import sys
import pandas as pd
from deepface import DeepFace
import warnings

# Suppress unnecessary warnings from deepface's backend
warnings.filterwarnings("ignore", category=UserWarning, module='deepface')

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "audit_results")
OUTPUT_CSV = os.path.join(BASE_DIR, "bias_audit_report.csv")

# --- Analysis Function ---
def analyze_image_attributes(image_path):
    """
    Analyzes a single image for demographic attributes using DeepFace.
    Returns a dictionary with the analysis results.
    """
    try:
        # The 'actions' parameter specifies which attributes to analyze
        # 'enforce_detection=False' attempts analysis even if face detection confidence is low
        analysis_result = DeepFace.analyze(
            img_path=image_path, 
            actions=['race', 'gender'],
            enforce_detection=False,
            silent=True # Suppresses verbose console output for each image
        )
        
        # DeepFace may return a dict (single face) or a list of dicts (multiple faces)
        if isinstance(analysis_result, list):
            first_face = analysis_result[0] if analysis_result else None
        else:
            first_face = analysis_result

        if first_face:
            return {
                'dominant_race': first_face.get('dominant_race', 'unknown'),
                'gender': first_face.get('dominant_gender', 'unknown')
            }
        else:
            return {'dominant_race': 'no_face_detected', 'gender': 'no_face_detected'}
            
    except Exception:
        # Handle cases where an image might be corrupted or cause an error
        return {'dominant_race': 'analysis_error', 'gender': 'analysis_error'}

# --- Main Loop ---
all_results = []
print(f"Starting analysis of images in '{INPUT_DIR}'...")

# Validate input directory exists
if not os.path.isdir(INPUT_DIR):
    print(f"Error: Input directory '{INPUT_DIR}' does not exist. Run 'run_audit.py' first.")
    sys.exit(1)

# Walk through the directory structure created by the generation script
for category in os.listdir(INPUT_DIR):
    category_path = os.path.join(INPUT_DIR, category)
    if not os.path.isdir(category_path):
        continue
        
    for prompt_folder in os.listdir(category_path):
        prompt_folder_path = os.path.join(category_path, prompt_folder)
        if not os.path.isdir(prompt_folder_path):
            continue
            
        prompt_text = prompt_folder.replace("_", " ")
        print(f"\nAnalyzing prompt: '{prompt_text}'")
        
        for image_file in os.listdir(prompt_folder_path):
            if image_file.lower().endswith(".png"):
                image_path = os.path.join(prompt_folder_path, image_file)
                
                # Analyze the image
                attributes = analyze_image_attributes(image_path)
                
                # Combine results into a single record
                record = {
                    'category': category,
                    'prompt': prompt_text,
                    'image_path': image_path,
                    **attributes  # Add the analyzed attributes
                }
                all_results.append(record)

# --- Save to CSV ---
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAnalysis complete. Report saved to '{OUTPUT_CSV}'")
else:
    print("\nNo images found to analyze.")