import csv

def extract_csv_text(file):
    """Extract text from CSV file"""
    all_text = []
    try:
        with open(file, 'r', encoding='utf-8') as file:
            # Try to detect delimiter
            sample = file.read(1024)
            file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            reader = csv.reader(file, delimiter=delimiter)
            for row_index, row in enumerate(reader):
                if row:  # Skip empty rows
                    row_text = ' | '.join([cell.strip() for cell in row if cell.strip()])
                    if row_text:
                        all_text.append(f"Row {row_index + 1}: {row_text}")           
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        with open(file, 'r', encoding='latin-1') as file:
            sample = file.read(1024)
            file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            reader = csv.reader(file, delimiter=delimiter)
            for row_index, row in enumerate(reader):
                if row:
                    row_text = ' | '.join([cell.strip() for cell in row if cell.strip()])
                    if row_text:
                        all_text.append(f"Row {row_index + 1}: {row_text}")
    except Exception as e:
        print(f"Error processing CSV: {e}")
    return '\n\n'.join(all_text)