
def readfile(file_info):
    pathname = file_info['path']
    if file_info['type'] == 'text/plain':
        with open(pathname, 'r') as file:
            data = file.read()
        return data
    if file_info['type'] == 'application/pdf':
        pdf_text = ""
        try:
            fd = open(pathname, "rb")
            viewer = SimplePDFViewer(fd)
            pdf_text = viewer.render()
        except Exception as e:
            print(f"Error reading PDF file: {e}")
        return pdf_text
    

def maketags(file_info):
    substraction = 'C:/SyncedFolder/Team Shares/FREA/'
    pathname = file_info['path'] 
    tagstring = pathname.replace(substraction, '')
    tagstring2 = tagstring.replace(file_info['name'], '')
    tags = tagstring2.split('/')
    print(tags)
    return tags

def process_file_getinfo(file_info):
    return_data = {}
    #data =readfile(file_info)
    tags = maketags(file_info) 
    #return_data['data'] = data
    #return_data['tags'] = tags

    return tags


def functext(file_info):
    process_file_getinfo(file_info)


#text/html
def funcWebPages(file_info):
    process_file_getinfo(file_info)

 #       'text/markdown': 
def funcMarkdown(file_info):
    process_file_getinfo(file_info)

 #       'application/xml': 
def funcXML(file_info):
    process_file_getinfo(file_info)

 #       'application/pdf': 
def funcPDF(file_info):
    process_file_getinfo(file_info)

 #       'application/msword': 
def funcDOC(file_info):
    process_file_getinfo(file_info)

 #       'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 
def funcDOCX(file_info):
    process_file_getinfo(file_info)

  #      'application/vnd.ms-excel (XLS)':
def funcXLS(file_info):
    process_file_getinfo(file_info)

 #       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
def funcXLSX(file_info):
    process_file_getinfo(file_info)

 #       'application/vnd.ms-powerpoint (PPT)':
def funcPPT(file_info):
    process_file_getinfo(file_info)

 #       'application/vnd.openxmlformats-officedocument.presentationml.presentation':
def funcPPTX(file_info):
    process_file_getinfo(file_info)

 #       'application/rtf':
def funcRTF(file_info):
    process_file_getinfo(file_info)

 #       'image/jpeg':
def funcJPG(file_info):
    process_file_getinfo(file_info)

  #      'image/png':
def funcPNG(file_info):
    process_file_getinfo(file_info)

 #       'image/gif': 
def funcGIF(file_info):
    process_file_getinfo(file_info)

#        'image/bmp': 
def funcBMP(file_info):
    process_file_getinfo(file_info)

 #       'image/tiff':
def funcTIFF(file_info):
    process_file_getinfo(file_info)

   #     'application/javascript': 
def funcJavaScript(file_info):
    process_file_getinfo(file_info)

  #      'application/zip': 
def funcZIP(file_info):
    process_file_getinfo(file_info)

  #      'application/gzip': 
def funcGZIP(file_info):
    process_file_getinfo(file_info)

  #      'audio/mpeg': 
def funcMP3(file_info):
    process_file_getinfo(file_info)

#        'video/mp4': 
def funcMP4(file_info):
    process_file_getinfo(file_info)

    #    'audio/wav': 
def funcWAV(file_info):
    process_file_getinfo(file_info)

 #       'audio/ogg': 
def funcOGG(file_info):
    process_file_getinfo(file_info)

  #      'video/webm': 
def funcWEBM(file_info):
    process_file_getinfo(file_info)

 #       'application/json': 
def funcJSON(file_info):
    process_file_getinfo(file_info)

 #       'application/x-yaml': 
def funcYAML(file_info):
    process_file_getinfo(file_info)

 #       'application/epub+zip': 
def funcEPUB(file_info):
    process_file_getinfo(file_info)

 #       'application/x-mobipocket-ebook': 
def funcMOBI(file_info):
    process_file_getinfo(file_info) 

def funcnone(file_info):
    process_file_getinfo(file_info) 


    
    
    