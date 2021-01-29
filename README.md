# Real time texture analysis for gastroscopic-exam
## Motivation
This project is developped in the scope of the PROJ-H402 course (yearly computing project) and is supervized by Pr. O.Debeir.

The goal of this project is to develop and implement a real time, autonomous, texture analyser for gastroscopic-exam for the service *Hépato-Gastro-Entérologie* of the *CHU
St-Pierre* in Brussels.

The main need of the medical team is to extract a standardized score based on the quantity of foam pollution in the gastroscopic videos. First, for use 
in the context of a medical study and then for 'everyday' physician assistance during gastroscopic exams.
## Language and libraries
Python is the chosen language to write the script due to its versatility. This project make an intensive use of the OpenCV library. 

## Functionning
The script reads a video flux from the path given in argument. (most common formats supported : *.avi, .mov, .mpeg, ...*) 

Each frame is treated independly : 

1. Determination of a uniformity score based on gaussian blur 
2. Foam segmentation in the HSV space
3. Detection enhancement with morphological transforms
4. Scoring based on pollution density
