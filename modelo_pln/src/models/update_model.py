import os
import shutil
import stat
import sys
from pathlib import Path

def check_file_permissions(filepath):
    """Check if we have write permissions for a file."""
    if not filepath.exists():
        return True  # File doesn't exist, we can create it
    return os.access(filepath.parent, os.W_OK) and os.access(filepath, os.W_OK)

def make_file_writable(filepath):
    """Make a file writable."""
    if not filepath.exists():
        return True
    try:
        current_perms = stat.S_IMODE(filepath.lstat().st_mode)
        os.chmod(filepath, current_perms | stat.S_IWUSR)
        return True
    except Exception as e:
        print(f"Warning: Could not make {filepath} writable: {e}")
        return False

def update_model():
    """Update the main model with the enhanced version."""
    # Define paths
    model_dir = Path("/home/sublime-dev/dev/python/Almabot/modelo_pln/data/models/")
    
    # Enhanced model files
    enhanced_model = model_dir / "enhanced_bullying_model.joblib"
    enhanced_vectorizer = model_dir / "enhanced_vectorizer.joblib"
    
    # Main model files (to be replaced)
    main_model = model_dir / "bullying_detection_model.joblib"
    main_vectorizer = model_dir / "vectorizer.joblib"
    
    # Check if enhanced model exists
    if not enhanced_model.exists() or not enhanced_vectorizer.exists():
        print("Error: Enhanced model files not found. Please train the enhanced model first.")
        return False
    
    # Check permissions
    if not check_file_permissions(main_model) and not make_file_writable(main_model):
        print(f"Error: No write permission for {main_model}")
        print("Please run this script with appropriate permissions or fix file permissions.")
        return False
        
    if not check_file_permissions(main_vectorizer) and not make_file_writable(main_vectorizer):
        print(f"Error: No write permission for {main_vectorizer}")
        print("Please run this script with appropriate permissions or fix file permissions.")
        return False
    
    try:
        # Create backup of current model if it exists
        if main_model.exists():
            backup_model = model_dir / "bullying_detection_model.joblib.bak"
            try:
                if backup_model.exists():
                    backup_model.unlink()
                shutil.copy2(main_model, backup_model)
                print(f"Created backup of current model at {backup_model}")
            except Exception as e:
                print(f"Warning: Could not create backup of current model: {e}")
        
        if main_vectorizer.exists():
            backup_vectorizer = model_dir / "vectorizer.joblib.bak"
            try:
                if backup_vectorizer.exists():
                    backup_vectorizer.unlink()
                shutil.copy2(main_vectorizer, backup_vectorizer)
                print(f"Created backup of current vectorizer at {backup_vectorizer}")
            except Exception as e:
                print(f"Warning: Could not create backup of current vectorizer: {e}")
        
        # Replace with enhanced model
        print("\nUpdating model files...")
        try:
            shutil.copy2(enhanced_model, main_model)
            shutil.copy2(enhanced_vectorizer, main_vectorizer)
            
            # Ensure the files are readable by the application
            os.chmod(main_model, 0o644)
            os.chmod(main_vectorizer, 0o644)
            
            print("\nModel updated successfully!")
            print(f"New model: {enhanced_model}")
            print(f"New vectorizer: {enhanced_vectorizer}")
            
            # Verify the update
            if main_model.exists() and main_vectorizer.exists():
                print("\nVerification: New model and vectorizer are in place.")
                return True
            else:
                print("Error: Failed to verify the update. Some files might be missing.")
                return False
                
        except Exception as e:
            print(f"Error during model update: {e}")
            print("Attempting to restore from backup...")
            # Try to restore from backup if available
            if 'backup_model' in locals() and backup_model.exists():
                try:
                    shutil.copy2(backup_model, main_model)
                    print("Restored model from backup.")
                except Exception as restore_error:
                    print(f"Failed to restore model from backup: {restore_error}")
            return False
            
    except Exception as e:
        print(f"Error updating model: {str(e)}")
        return False

if __name__ == "__main__":
    update_model()
