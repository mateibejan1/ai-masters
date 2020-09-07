Chenges made to the code:

1.
Because the initial bounding boxes' format was changed for task 3 (the bounding box is now on line 3 instead of line 2), I had to change the code inside the track_white_ball and track_coloured_ball functions. What changed was the initialization of the 1st frame's bounding boxes. 

So the following code:

curr_bbox = (int(lines[-1].split(' ')[1]), 
		 int(lines[-1].split(' ')[2]), 
		 int(lines[-1].split(' ')[3]) - int(lines[-1].split(' ')[1]), 
		 int(lines[-1].split(' ')[4]) - int(lines[-1].split(' ')[2]))

turned to this in both functions:

curr_bbox = (int(lines[-2].split(' ')[1]), 
		 int(lines[-2].split(' ')[2]), 
		 int(lines[-2].split(' ')[3]) - int(lines[-2].split(' ')[1]), 
		 int(lines[-2].split(' ')[4]) - int(lines[-2].split(' ')[2]))

2.
I had to remove the task1_ground_truth_paths array from the code that solves task 1 (the cell right below the cell containing the detect_snooker_balls function). The array was there so that I could compare the results with the ground truths for the train set and I have forgotten to remove it in the first version of the project.

3.
There was a bug in potted_ball_detection in which a white_ball_pos would remain empty after assignment because it there might be issues with detecting it in the first few frames. This bug did not appear when the code was run on the training data, so I had to patch it now. 

Please note that the patch DOES NOT change the ball detection algorithm or the pot detection algorithm. The ccs array contains the bounding boxes for all balls for each frame. Instead of taking the bounding box for the cue ball from the first frame, I took the bounding box from the first frame in which the white ball is detected.

Again, this did not come up as a problem for the training data.
