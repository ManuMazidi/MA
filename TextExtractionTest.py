import PyPDF2 as pdf
import re
import glob
from datetime import datetime



file_paths = glob.glob('/Users/manu/Documents/MA/Data/*')



for filename in file_paths:

    # get the date of the file
    find_date = re.search("[0-9]{8}", filename)
    pdf_date = datetime.strptime(find_date.group(), '%Y%m%d')
    pdf_datetime = pdf_date.date()
    print(pdf_datetime)

    # read PDF
    pdf_file = open(filename, 'rb')
    read_pdf = pdf.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()

    # get number of pages per file
    print(number_of_pages)

    # get text
    page_content = ""  # define variable for using in loop.
    for page_number in range(number_of_pages):
        page = read_pdf.getPage(page_number)
        page_content += page.extractText()  # concate reading pages.

    print(page_content)

