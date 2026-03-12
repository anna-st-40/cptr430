# Names: Jaime Garcia, Anna Stefaniv Oickle
# Lab: lab7 (Vision)
# Date: March 12

**Question 1:** Describe two or three distinct visual patterns you observe across the 16 feature maps. What kind of image structures (edges, textures, colors) does Layer 1 appear to be responding to?

Filter 4 seems to be detecting the background with a very light color. Filter 11, 8 shows a lot of texture of the animal. Layer 1 appears to be responding to pretty much everything mentioned in the question statement (edge, texture, colors) but in different feature maps.

**Question 2:** Some feature maps appear almost entirely dark or uniform. What does a near-zero activation mean in terms of what the filter "found" in this particular image? What would cause a filter to activate strongly?

The filter didn't find what it was looking for in that image. The filter would activate upon finding signs or patterns of what it's looking for.

**Question 3:** Q3. How do the Layer 4 feature maps differ visually from Layer 1 maps? What does this tell you about the nature of the representations learned at different depths?

Layer 4 has much smaller resolution and we can barely identify any object or subject. Layer 1 is more detailed and deep and layer 4 is more shallow.

**Question 4:** If you used a completely different image (e.g., a car instead of a dog), which layer's feature maps would change more dramatically — Layer 1 or Layer 4? Justify your answer based on what each layer has learned.

Layer 1 would change more dramatically because in layer 1 you are actually able to identify the object or subject whereas in layer 4, you will get the same feel or sense of the image, squares, because layer 4 is shallow and layer 1 is deep.

**Question 5:** Describe the spatial pattern of the Grad-CAM heatmap. Does the model appear to focus on the object of interest or on surrounding context? What are the implications for model trustworthiness?

The model appears to be focused on the object of interest. For the model to be trustworthy, if it focus on what we want it to focus on then it's trustworthy.

**Question 6:** Grad-CAM uses gradients from a specific class to generate the heatmap. What would happen to the heatmap if you asked it to explain the prediction for a wrong class (e.g., generating a heatmap for "cat" when the image contains a dog)?

We don't think it would do very well because it would get confused as it doesn't recognize the subject requested. The model would focus more on the parts of the image that look similar to those of a cat, even though the real image is a dog

**Question 7:** If a model correctly classifies a "polar bear" image but Grad-CAM shows it attended primarily to the snow background, what does this tell us about what the model actually learned? How might this model perform on a polar bear image in a zoo?

If there is snow, the animal is a polar bear probably, because it learned that if there's snow then the animal will be a polar bear. We don't think it would perform well on an image of a polar bear in a zoo because it learned that snow is associated with polar bears, maybe it would recognize it as an animla the size of a polar bear.

**Question 8:** Look at the amplified perturbation image. Does it look like anything meaningful to you? Now consider that the model finds this pattern highly informative — what does this reveal about the difference between how humans and CNNs represent images?

It doesn't look like anything meaningful to us. The prediction of the model changes to a wood rabbit and is highly confident about it with 99%. The CNN interprets individual pixels that we are almost unable to see the difference, and maybe those pixels contain information that we are unable to see.

**Question 9:** At what epsilon value does the model lose confidence in the correct class below 50% in your run? What does this tell you about how close the original image is to the decision boundary in pixel space?

The confidence value drops at $\epsilon$ = 0.005. It's very close, you only need to chagne or add a very very tiny adversarial perturbation to mess the model up.

**Question 10:** Consider a safety-critical application like medical imaging diagnosis or autonomous driving. Based on your observations across Exercises 6 and 7, what specific risks do adversarial examples pose, and what would you want to know about a deployed model's adversarial robustness before trusting it?

If it's safety-critical, we would want the model to be adversarially-robust, we wouldn't want the model to collapse upon a tiny amount of perturbation. We would want the model to have a higher $\epsilon$ at which it loses confidence