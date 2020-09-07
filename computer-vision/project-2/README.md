# Computer Vision Project 2: Video Analysis of Snooker Footage

1. Library versions:

opencv_python==4.2.0
numpy==1.18.1

2. How to run each scenario and where to look for the output file:

All output files are contained in a folder named "output", which is part of the archive. Each file is placed in its tasks' folder, so paths will be of form "output/Task1/1.txt". 

The .txt files containing the predicted data are named accordingly to the naming conventions. Since an output format has not been provided for task 4, we took the liberty of naming them similarly to those in task 3 (the first video will have an output file called "1_ball_1.txt", the second one "2_ball_1.txt" and so on. 

The project file is a jupyter notebook, so in order for the scenarios/tasks to run, the user must run all cells in the "Preliminaries", "Define utility functions" and all functions defined in each tasks' section. If the user does not have OpenCV installed on his machine, they can uncomment the 2nd cell in the project in order to install OpenCV 4.2.0 via pip.

- Scenario 1: Run the cells in the "Task 1" section. The output text files will be written inside the "Task1" folder.

- Scenario 2: Run the cells in the "Task 2" section. The output text files will be written inside the "Task2" folder.

- Scenario 3: Run the cells in the "Task 3" section. The output text files will be written inside the "Task3" folder.

- Scenario 4: Run the cells in the "Task 4" section. The output text files will be written inside the "Task4" folder.

Task1, Task2, Task3 adn Task4 folder must be manually added in the outputs folder by the user.

The only things the user will need to change are the paths inside the cell in the "Preliminaries" section. The paths must lead to the folders where the user has stored the images corresponding to each task, followed by a "*.jpg" for images, "*.mp4" for videos and "*.txt" for text files. The paths are all loaded in the 3rd cell of the notebook.
