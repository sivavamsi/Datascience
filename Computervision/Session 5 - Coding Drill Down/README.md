# Model 1

 # **Set up:**
* Set Transforms, Data Loader.
* Set Basic Training  & Test Loop
* Used 2 FC's to bring it down to 10 outputs.
 #**Results:**
* Parameters: 1M  (1,081,282)
* Best Training Accuracy: 99.85
* Best Test Accuracy: 99.04
  #**Analysis:**
* Extremely Heavy Model for such a problem
* Not a bad model, as there is not much over fitting. Loss seems to be decreasing steadily.

# Model 2

  # **Set up:**
* Replace FC with 1x1(equivalent to FC but conv) followed by GAP. (hugely reduces params)
* Reduce the number of kernels. (reduces the params)
 #**Results:**
* Parameters: 9,680
* Best Training Accuracy: 98.71
* Best Test Accuracy: 98.27
  #**Analysis:**
* Extremely Heavy Model for such a problem
* Not a bad model, as there is not much over fitting. Loss seems to be decreasing steadily.


# Model 3

  # **Set up:**
* Add Batch Norm.
* Add some more capacity.
  #**Results:**
* Parameters: 9,142
* Best Training Accuracy: 99.26
* Best Test Accuracy: 99.21
 #**Analysis:**
* Not a bad model, as there is not much over fitting. Loss seems to be decreasing steadily.

# Model 4

  # **Set up:**
* Changed structure, moved maxpool to after 4 th conv.
* Added dropout with very small capacity
  #**Results:**
* Parameters: 9,142
* Best Training Accuracy: 99.22
* Best Test Accuracy: 99.23
  #**Analysis:**
* better than train accuracy ,test accuracy is good
