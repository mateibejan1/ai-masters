# Computer Vision Project 1: Automatic Grading of Multiple Choice Tests

1. Library versions:

opencv_python==4.1.0
numpy==1.18.1
keras==2.2.4-tf

2. How to run each scenario and where to look for the output file:

All output files are contained in a folder named "output", which is part of the archive. The .txt files containing the predicted data are named accordingly to the naming conventions.

Since the project file is a jupyter notebook, in order for the scenarios/tasks to run, the user must load all the cells in the "Preliminaries" and "Define utility functions" sections. The actual code to solve the tasks can be found in the "Implementation" section. 

- Scenario 1: Run the cell in the "1. Real world" section. The output will be written inside the matei_bejan_407_task1.txt file.

- Scenario 2: Run the two cells in the "2. Intermediate" section. The first will parse all rotated pcitures and the second, all the perspective pictures. The output will be written inside the matei_bejan_407_task2.txt file.

- Scenario 3: Run the cell just in the "3. No Annotations" section. The output will be written inside the matei_bejan_407_task3.txt file.

- Scenario 4: Run the cell just in the "4. Handwritten recognition" section. The output will be written inside the matei_bejan_407_task4.txt file.

The only things the user will need to change are the paths inside the cell in the "Load data" subection, which is part of the "Preliminaries". The paths must lead to the folders where the user has stored the images corresponding to each task.
