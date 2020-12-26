def handleStrData(strData):
    if len(strData)<=2:
        return strData 
    else:
        p_n_1 = 0
        flag = None
        strList = []
        
        i =0
        while i<len(strData):
            # print(i,flag)
            if flag is not None:
                char = strData[i]
                strList.append(strData[i])
                j = i +1
                while j<len(strData) and char == strData[j]:
                    j +=1
                # print(i,j)
                i = j
                flag = None
            else:
                print('strList',strList)
                char = strData[i]
                j = i +1
                strList.append(strData[i])
                while j<len(strData) and char == strData[j]:
                    j +=1
                print(i,j,char)
                if j>=i+2:
                    strList.append(strData[i+1])
                    flag = strData[i+1]
                print('strList',strList)
                i = j
    return ''.join(strList)

print(handleStrData('wooooow'))