"""
PDF utility functions for processing and analyzing PDF files.
"""
import fitz  # PyMuPDF
from pathlib import Path
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def pdf_has_images(pdf_path: str) -> bool:
    """
    Check if a PDF file contains any images.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        bool: True if the PDF contains images, False otherwise
    """
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            if len(image_list) > 0:
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking PDF for images: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def convert_pdf_to_markdown(pdf_path: str) -> Tuple[str, str]:
    """
    Convert a PDF file to Markdown format.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Tuple[str, str]: A tuple containing (markdown_content, title)
    """
    try:
        doc = fitz.open(pdf_path)
        markdown_content = []
        title = Path(pdf_path).stem
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Add page break marker
            if page_num > 0:
                markdown_content.append("\n\n---\n\n")
            
            # Process text blocks
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4]  # The actual text content is at index 4
                
                # Basic formatting detection (very simple heuristic)
                if len(text.strip()) > 0:
                    first_line = text.strip().split('\n')[0]
                    if len(first_line) < 100 and first_line.isupper():  # Likely a header
                        markdown_content.append(f"## {first_line}\n")
                    else:
                        markdown_content.append(text.strip() + "\n\n")
        
        return "\n".join(markdown_content), title
    except Exception as e:
        logger.error(f"Error converting PDF to Markdown: {str(e)}")
        raise
    finally:
        if 'doc' in locals():
            doc.close()

def read_pdf_bytes(pdf_path: str) -> bytes:
    """
    Safely read PDF file as bytes with proper error handling.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        bytes: Raw bytes of the PDF file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there are issues reading the file
    """
    try:
        with open(pdf_path, 'rb') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
    except IOError as e:
        raise IOError(f"Error reading PDF file {pdf_path}: {str(e)}")
