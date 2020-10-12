import PyPDF2 as pdf
import glob

file_paths = glob.glob('/Users/manu/Documents/MA/Data/*')

for i in file_paths:
    pdf_file = open(i, 'rb')
    read_pdf = pdf.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    print(number_of_pages)

    page_content = ""  # define variable for using in loop.
    for page_number in range(number_of_pages):
        page = read_pdf.getPage(page_number)
        page_content += page.extractText()  # concate reading pages.

    print(page_content)
