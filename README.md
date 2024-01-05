# About

This project contains the code that was used for training and inference of the model. It also contains the model weights in `DigitRecognition.h5`.

I used pygame to display a canvas where the user could draw a digit and the model would try predicting the digit. Given the intended digit, the model will then train
on the drawn image if it predicted incorrectly.

This is all run in a recurring loop which effectively means that this is a self-learning model that constantly fits on new digit images.

## Technical Considerations

When a user draws an digit, the model is trained on that specific digit 100 times. This is **not** the best approach to self-learning. The model should see varied and diverse
samples in order to better generalize the handwritten digits and not overfit to any one specific image. However, a person's handwriting is often consistent, which would mean
overfitting would be desired. The model should fit exactly to the style of a user's handwriting to be really good at classifying it. Of course, if a user's handwriting is not consistent
overfitting would result in poor accuracy. Finding a balance is necessary.
