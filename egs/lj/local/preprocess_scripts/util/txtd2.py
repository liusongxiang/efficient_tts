import argparse
import os
import sys    
    
def label_transformation(args) :
    f = open(args.input)           
    h = open(args.input2)
    g = open(args.output, 'w')

    line = f.readline() 
    num1=0
    while line:
        line = f.readline()         
        num1 = num1 + 1
    
    
    line = h.readline()
    num2=0
    while line:
        line = h.readline()
        num2 = num2 + 1

    if num1 != num2 :
        f.close()
        h.close()
        exit()
    
    f.seek(0)
    h.seek(0)

    line = f.readline()
    lineh = h.readline()
    while line:
        print line,               
        line=line.strip('\r\n')
        time1,dur = line.split()
        
        print(time1)
        print(dur)
        time1x = int(float(time1)* 10000000 + 0.5)
        time2x = int((float(time1)+float(dur))* 10000000 + 0.5)
        
        print(time1x)
        print(time2x)
        lineh=lineh.strip('\r\n')
        x,y,label = lineh.split()
        
        linez = str(time1x) + " " + str(time2x) + " " + label
        #pos = line.find('-')
        #linez = line[:pos]
        #pos1 = line.find('+',pos+1)
        #linex = line[pos+1:pos1]
        g.writelines(linez)
        #g.writelines(linex)
            
        line = f.readline()
        lineh = h.readline()
        g.writelines("\n")

    f.close()
    h.close()
    g.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_calculate_mixture_features = subparsers.add_parser('txtd')
    parser_calculate_mixture_features.add_argument('--input', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--input2', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--output', type=str, required=True)
    
    ### command: python add_white_noise.py add_noise --workspace="ws_addnoise" --speech_dir="test_wav" --snr=30
    args = parser.parse_args()
    if args.mode == 'txtd':
        label_transformation(args)
    else:
        raise Exception("Error!")
