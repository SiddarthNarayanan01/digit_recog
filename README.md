# About

This project contains the code used for training and inference of the model. It also includes the model weights in `DigitRecognition.h5`.

I used Pygame to display a canvas where the user could draw a digit, and the model would try to predict the digit. Given the intended digit, the model will then train
on the drawn image if it is mispredicted.

This is all run in a recurring loop, which effectively means that this is a self-learning model that constantly fits on new digit images.

## Technical Considerations

When a user draws a digit, the model is trained on that specific digit 100 times. There are better approaches to self-learning than this. The model should see varied and diverse
samples to generalize the handwritten digits better and not overfit to any one specific image. However, a person's handwriting is often consistent, meaning
overfitting would be desired. The model should fit the style of a user's handwriting to be good at classifying it. Of course, if a user's handwriting is inconsistent,
overfitting would result in poor accuracy. Finding a balance is necessary.
