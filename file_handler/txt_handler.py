def extract_txt_text(file):
    """Extract text from TXT file"""
    try:
        with open(file, 'r', encoding='utf-8') as file:
            content = file.read()
        return content.strip()
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        with open(file, 'r', encoding='latin-1') as file:
            content = file.read()
        return [content.strip()] if content.strip() else []
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""