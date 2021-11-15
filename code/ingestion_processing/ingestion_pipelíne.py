# set for ocr
def get_tools():
    import glob
    import os
    import shutil
    import requests

    file_set = set(glob.glob("*"))
    if "gs" not in file_set: # gs
        gs_url = "https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs9540/ghostscript-9.54.0-linux-x86_64.tgz"
        r = requests.get(gs_url, allow_redirects=True)
        open("gs.tgz", "wb").write(r.content)
        os.system("tar -xzvf gs.tgz")
        os.remove("gs.tgz")
        shutil.move("ghostscript-9.54.0-linux-x86_64/gs-9540-linux-x86_64","gs")
        shutil.rmtree("ghostscript-9.54.0-linux-x86_64", ignore_errors=True)
    if "eng.traineddata" not in file_set: # lstm
        tes_url = "https://github.com/tesseract-ocr/tessdata_best/raw/master/eng.traineddata"
        r = requests.get(tes_url, allow_redirects=True)
        open("eng.traineddata", "wb").write(r.content)

# load input data
def get_meta(path):
    import pandas as pd

    def sanitize_names(r):
        import re

        ind = str(r.name).rjust(4,"0")
        year = str(r.Year)
        part = r.Participant
        part = part.lower()
        part = re.sub("[.!?\\-/,\"\(\)\+'\|]"," ",part)
        part = re.sub("&"," and ",part)
        part = re.sub(" +","-",part)
        part = re.sub("-$","",part)
        pp = [ind, year, part]
        res = "-".join(pp)
        if len(r.FileType)>0:
            res = res+"."+r.FileType
        return res

    docs = pd.read_excel("../../data/un-global-impact.xlsx", sheet_name=[0,1])
    docs = pd.concat([v.loc[:,"Participant":"EconomyType"] for k,v in docs.items()])
    # get only the eng ones with a link
    docs = docs[(docs.Language=="english") & (docs.Link.notnull())].\
        sort_values(["Year", "Participant", "Link"]).drop_duplicates("Link").\
            reset_index(drop=True)
    # file types
    file_type = docs.Link.str.split("/").apply(lambda x: x[-1].split("?")[0].split(".")[-1]).str.lower()
    docs["FileType"] = ["" if f not in set(["pdf", "docx", "pptx", "doc", "docm", "htm", "html"]) else f\
        for f in file_type.values]
    docs["FileName"] = docs.apply(sanitize_names, axis=1)
    return docs

# download papers       
def download_docs(docs, path):
    import os
    import requests
    import datetime
    import concurrent.futures

    def download_doc(ind=0, df=docs, output_path=path):
        r = requests.get(df.loc[ind].Link, allow_redirects=True)
        open(output_path+df.loc[ind].FileName, "wb").write(r.content)

    # par-all
    st = datetime.datetime.now()
    if not os.path.exists(path):
        os.mkdir(path)    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe: 
        exe.map(download_doc,  docs.index) # docs.index
    td = (datetime.datetime.now()-st).seconds/60
    
    # NOTE ADD STATUS CHECK
    print("The download took {:.2f} mins on the threadpool back-end.".format(td))

def unify_format(docs, path):
    import magic
    import numpy as np
    import datetime
    import os

    docs["FileTypeMagic"] = docs.FileName.apply(lambda x: magic.from_file(path+x,
        mime=True))
    st = datetime.datetime.now()
    docs["ConversionStatus"] = np.nan
    docs["ConvertedFileName"] = docs["FileName"] # prefill
    docs_to_convert_index = docs.loc[(docs.FileTypeMagic!="inode/x-empty") & (docs.FileTypeMagic!="application/pdf")].index
    for i in docs_to_convert_index:
        input_file = path+docs.loc[i].FileName
        output_file = path+docs.loc[i].FileName.split(".")[0]+".pdf"
        try:
            cmd = "unoconv -f pdf -T 30 -o "+output_file+" "+input_file
            docs.loc[i,"ConversionStatus"] = os.system(cmd)
            docs.loc[i,"ConvertedFileName"] = output_file       
        except:
            print("Conversion of the file {} failed.".format(input_file))
            docs.loc[i,"ConvertedFileName"] = np.nan   
            continue
        os.system("pkill soffice.bin")
    td = (datetime.datetime.now()-st).seconds/60
    pdf_count = np.sum(docs.ConversionStatus==0)
    pdf_remainder_count = len(docs_to_convert_index)-pdf_count
    print("Type conversion finished in {:.2f} mins with {} pdf files and the remainder of {} raw files.".\
        format(td, pdf_count, pdf_remainder_count,))
    return docs

def do_ocr(docs, txt_path, pdf_path):
    import os
    import datetime
    import numpy as np
    import pandas as pd

    def pdf2txt(row, txt_path, pdf_path):
        import os
        import numpy as np
        
        input_file = pdf_path+row.ConvertedFileName
        output_file =  txt_path+row.FileName.split(".")[0]+".txt"
        # try to ocr the files
        try:
            cmd = "./gs -sDEVICE=ocr -r200 -dQUIET -dBATCH -dNOPAUSE -sOutputFile="+\
                output_file+" "+input_file
            row["OCRStatus"] = os.system(cmd)
            row["OutputFileName"] = output_file
            with open(output_file, "r") as of:
                row["OutputText"] = of.read().replace("\n", " ")
        except:
            print("OCR of the file {} failed.".format(input_file))
            row["OutputFileName"] = np.nan
            return row
        return row    

    st = datetime.datetime.now()
    if not os.path.exists(pdf_path):
        os.mkdir(pdf_path)   
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    docs["OCRStatus"] = np.nan
    docs["OutputText"] = np.nan
    docs["OutputFileName"] = np.nan
    from joblib import Parallel, delayed
    temp_docs = Parallel(n_jobs=6, backend="loky")(
        delayed(pdf2txt)(docs.loc[i,:], txt_path, pdf_path) for i in docs.index)
    docs = pd.concat(temp_docs, axis=1).transpose()    
    td = (datetime.datetime.now()-st).seconds/60
    text_count = docs.shape[0]
    text_remainder_count = docs.OutputText.isna().sum()
    print("Text conversion finished in {:.2f} mins with {} pdf files and the remainder of {} raw files.".\
        format(td, text_count-text_remainder_count, text_remainder_count))
    return docs    

if __name__=="__main__":
    raw_path, txt_path = "../../data/raw_reports/", "../../data/txt_reports/"

    import shutil
    shutil.rmtree(raw_path, ignore_errors=True)
    shutil.rmtree(txt_path, ignore_errors=True)
    get_tools()
    docs = get_meta("../../data/un_global_impact.xlsx")
    #docs = docs.iloc[:250,:]
    download_docs(docs, raw_path)
    docs = unify_format(docs, raw_path)
    docs = do_ocr(docs, txt_path, raw_path)
    docs.to_pickle("../../data/texts.pkl")
