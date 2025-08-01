from pypdf import PdfWriter
import os

# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# List of all your PDF filenames, including the new one
pdf_list = [
    "BAJHLIP23020V012223.pdf",
    "CHOTGDP23004V012223.pdf",
    "EDLHLGA23009V012223.pdf",
    "HDFHLIP23024V072223.pdf",
    "ICIHLIP22012V012223.pdf",
    # Add the new PDF filename here
    "Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf",
]

# The output merged PDF will be in the main project root
output_filename = os.path.join(script_dir, 'policy.pdf')

merger = PdfWriter()
print("Starting PDF merge...")

# Iterate through all the PDFs
for pdf_file in pdf_list:
    # Construct the full path to the source PDF within 'source_pdfs'
    full_pdf_path = os.path.join(script_dir, 'source_pdfs', pdf_file)
    if os.path.exists(full_pdf_path):
        print(f"  -> Adding {pdf_file}...")
        merger.append(full_pdf_path)
    else:
        print(f"  -> WARNING: {full_pdf_path} not found. Skipping.")

# Write out the merged PDF
with open(output_filename, "wb") as fout:
    merger.write(fout)

merger.close()
print(f"\n✅ Successfully merged files into '{os.path.abspath(output_filename)}'!")
print("This 'policy.pdf' will be used by the FastAPI application.")
