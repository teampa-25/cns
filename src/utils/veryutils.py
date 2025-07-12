"""
Utility functions for creating ZIP archives of output images.
"""

import os
import zipfile
from pathlib import Path
import io
import shutil
import numpy as np


def clean_output_images(
    output_dir: str = "output_images",
    keep_subdirs: bool = False
) -> int:
    """
    Clean up the output images directory.
    
    Args:
        output_dir: Path to the output images directory
        keep_subdirs: If True, only delete files but keep directory structure
        
    Returns:
        int: Number of files deleted
    """
    
    if not os.path.exists(output_dir):
        return 0
    
    deleted_count = 0
    
    if keep_subdirs:
        # Only delete files, keep directory structure
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
    else:
        # Remove entire directory and recreate it
        import shutil
        try:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Cleaned directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean directory {output_dir}: {e}")
    
    return deleted_count


def create_zip_stream(
    request_id: str, 
    output_dir: str = "output_images",
    cleanup_after: bool = True
) -> bytes:
    """
    Create a ZIP archive of images for a specific request ID in memory.
    
    Args:
        request_id: The request ID subdirectory to zip
        output_dir: Path to the output images directory (default: "output_images")
        cleanup_after: Whether to clean up the output directory after zipping
        
    Returns:
        bytes: ZIP archive as bytes stream
        
    Raises:
        FileNotFoundError: If request directory doesn't exist
        ValueError: If no images found in the directory
    """
    
    request_dir = os.path.join(output_dir, request_id)
    
    if not os.path.exists(request_dir):
        raise FileNotFoundError(f"Request directory '{request_dir}' does not exist")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Collect image files in the request directory
    image_files = []
    for file in os.listdir(request_dir):
        file_path = os.path.join(request_dir, file)
        if os.path.isfile(file_path) and Path(file).suffix.lower() in image_extensions:
            image_files.append((file_path, file))
    
    if not image_files:
        raise ValueError(f"No image files found in '{request_dir}'")
    
    # Create ZIP archive in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, archive_name in image_files:
            zipf.write(file_path, archive_name)
            print(f"Added to ZIP: {archive_name}")
    
    # Get the ZIP bytes
    zip_bytes = zip_buffer.getvalue()
    zip_buffer.close()
    
    print(f"ZIP archive created in memory ({len(image_files)} images, {len(zip_bytes)} bytes)")
    
    # Clean up the output directory if requested
    if cleanup_after:
        try:
            shutil.rmtree(request_dir)
            print(f"Cleaned up directory: {request_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {request_dir}: {e}")
    
    return zip_bytes


def create_all_images_zip_stream(
    output_dir: str = "output_images",
    cleanup_after: bool = True
) -> bytes:
    """
    Create a ZIP archive of all images (in the output directory) in memory.
    
    Args:
        output_dir: Path to the output images directory (default: "output_images")
        cleanup_after: Whether to clean up the output directory after zipping
        
    Returns:
        bytes: ZIP archive as bytes stream
        
    Raises:
        FileNotFoundError: If output directory doesn't exist
        ValueError: If no images found in the directory
    """
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Collect all image files recursively
    image_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                # Calculate relative path for ZIP archive
                rel_path = os.path.relpath(full_path, output_dir)
                image_files.append((full_path, rel_path))
    
    if not image_files:
        raise ValueError(f"No image files found in '{output_dir}'")
    
    # Create ZIP archive in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path, archive_name in image_files:
            zipf.write(file_path, archive_name)
            print(f"Added to ZIP: {archive_name}")
    
    # Get the ZIP bytes
    zip_bytes = zip_buffer.getvalue()
    zip_buffer.close()
    
    print(f"ZIP archive created in memory ({len(image_files)} images, {len(zip_bytes)} bytes)")
    
    # Clean up the output directory if requested
    if cleanup_after:
        try:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Cleaned up directory: {output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {output_dir}: {e}")
    
    return zip_bytes



"""
Utility to recursively convert all numpy ndarrays to lists
Placed at module level to avoid UnboundLocalError.
"""
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray(i) for i in obj)
    else:
        return obj