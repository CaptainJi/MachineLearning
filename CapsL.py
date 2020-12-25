def test_caps(str,caps=False):
    i=0
    for word in str:
        if word.islower() == False and caps == False:
            i+=1
            caps=True
        if word.islower() == True and caps == True:
            i+=1
            caps=False
    print(i)

if __name__ == '__main__':
    str='ADFASsdfSADF'
    test_caps(str)