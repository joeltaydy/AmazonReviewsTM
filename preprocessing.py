import csv, json, os,time,traceback
rootdir= 'AmazonReviews'
outputFileDict = {
    "outputFileProductInfo" : 'data/preprocessed_productinfo.csv',
    "outputFileReviews" : 'data/preprocessed_reviewinfo.csv'
}

def readManualLabel(file):
    dictLabel = {}
    with open(file,encoding="utf-8",errors='ignore') as csvfile:
        readCSV = csv.reader(csvfile)
        next(readCSV, None)  # skip the headers
        for row in readCSV:
            dictLabel[row[3]] = row[7]
    return dictLabel

def readJsonData(rootdir):
    dataArray=[]
    for subdir, dirs, files in os.walk(rootdir):
        for file in files[:100]: 
            #print(os.path.join(subdir, file))
            with open(os.path.join(subdir, file)) as infile:
                jsonObj = json.loads(infile.read())
                jsonObj['category'] = subdir.split('\\')[1]
                dataArray.append(jsonObj) 
    return dataArray

def getHeaders(dataArray):
    header = {}
    for i in dataArray[0].keys():
        try:
            if(type(dataArray[0][i])==list):
                e= dataArray[0][i][0].keys()
            else:
                e = dataArray[0][i].keys()
            header[i] = list(e)
        except:
            pass
    return header

def getPolarity(overallScore):
    if overallScore <3 :
        return "0"
    else:
        return "1"

def writeToFile(dataArray, header,outputFileDict):
    dictLabel=readManualLabel('data/shit_hole.csv')
    for typeOfFile in header.keys():
        outputFile = "outputFile"+ typeOfFile
        with open(outputFileDict[outputFile],'w', newline='',encoding='utf-8') as outputFile:            
            f = csv.writer(outputFile)
            file_header = ['category']
            headers= header[typeOfFile]
            file_header.extend(headers)
            checker= []
            if type(dataArray[0][typeOfFile])==list:
                file_header.append('polarity')
            f.writerow(file_header)
            for data in dataArray[:500]:
                dataReview = [data['category']]
                try:
                    if type(data[typeOfFile])==list:
                        for review in data[typeOfFile]:
                            if review['ReviewID'] not in checker:
                                for subheader in headers:
                                    dataReview.append(review[subheader])
                                if float(review['Overall']) != 3:
                                    dataReview.append(getPolarity(float(review['Overall'])))
                                else: 
                                    try:
                                        dataReview.append(dictLabel[review['ReviewID']])
                                    except:
                                        continue
                                if None not in dataReview:
                                    if len(dataReview) == 8:
                                        f.writerow(dataReview)
                                    checker.append(dataReview) 
                                    checker.append(review['ReviewID'])
                                dataReview = [data['category']]
                                
                    else:
                        
                        for subheader in headers:
                            dataReview.append(data[typeOfFile][subheader])
                        if None not in dataReview:
                            f.writerow(dataReview)
                
                except Exception:
                   #pass
                    traceback.print_exc()


start = time.time()

dataArray = readJsonData(rootdir)
print("Finish reading the json data. Time Taken: " + str(time.time() - start))
header = getHeaders(dataArray)
writeToFile(dataArray, header,outputFileDict)
print("Finish preprocessing into csv. Time taken: " + str(time.time() - start)) 