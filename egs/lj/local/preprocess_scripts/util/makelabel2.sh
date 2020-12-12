#!/bin/bash

#fileo2=/home/bearlu/all_eng.txt
#for lab in ./labelfull/*.lab; do \
rm -rf ./mono
rm -rf ./newlabfull
mkdir ./mono
mkdir ./newlabfull

cat ./vocals.ali.phone.sym.ctm  | while read line
do
      name=`echo $line | awk '{print $1}'`
      time1=`echo $line | awk '{print $3}'`
      time2=`echo $line | awk '{print $4}'`
      base=`basename ${name}`; \
      fileo=./mono/${base}.lab
      
      time=${time1}" "${time2}
      echo $time >> $fileo
      #echo " " >> $fileo
      #echo $time2 >> $fileo
      #echo "\n" >> $fileo
      echo ${base} 
      #echo ${fileo} 
      #python ./txtd2.py txtd --input ${lab} --output ${fileo}
      #cmd="cat ${lab} >> ${fileo2}"
      #echo $cmd
      #$cmd
done ;


for lab in ./mono/*.lab; do \
      #name=`echo $line | awk '{print $1}'`
      #time1=`echo $line | awk '{print $3}'`
      #time2=`echo $line | awk '{print $4}'`
      base=`basename ${lab} .lab`; \
      lab1=./full-with-dur/${base}.lab
      fileo=./newlabfull/${base}.lab
      
      echo ${base}
      #time=${time1}" "${time2}
      #echo $time >> $fileo
      #echo " " >> $fileo
      #echo $time2 >> $fileo
      #echo "\n" >> $fileo
      echo ${lab} 
      echo ${lab1} 
      echo ${fileo} 
      python ./txtd2.py txtd --input ${lab} --input2 ${lab1} --output ${fileo}
      #cmd="cat ${lab} >> ${fileo2}" 
      #echo $cmd
      #$cmd
 done ;
