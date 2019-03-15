import csv, json, os,time,traceback
rootdir= 'AmazonReviews'
outputFileDict = {
    "outputFileProductInfo" : 'preprocessed_productinfo.csv',
    "outputFileReviews" : 'preprocessed_reviewinfo.csv'
}

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
        return "-"
    else:
        return "+"

def writeToFile(dataArray, header,outputFileDict):
    for typeOfFile in header.keys():
        outputFile = "outputFile"+ typeOfFile
        with open(outputFileDict[outputFile],'w', newline='',encoding='utf-8') as outputFile:            
            f = csv.writer(outputFile)
            file_header = ['category']
            headers= header[typeOfFile]
            file_header.extend(headers) 
            if type(dataArray[0][typeOfFile])==list:
                file_header.append('polarity')
            f.writerow(file_header)
            for data in dataArray[:500]:
                dataReview = [data['category']]
                try:
                    if type(data[typeOfFile])==list:
                        
                        for review in data[typeOfFile]:
                            for subheader in headers:
                                dataReview.append(review[subheader])
                            dataReview.append(getPolarity(float(review['Overall'])))
                            if None not in dataReview:
                                f.writerow(dataReview)
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