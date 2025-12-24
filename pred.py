def pred_and_plot(model, filename,class_names=class_names):
  """
  imports an image located at filename , makesa prediction with model
  and plots the image with the predicted class at the title
  """

  img = load_and_prep_image(filename)

  pred = model.predict(tf.expand_dims(img,axis=0))

  pred_class = class_names[int(tf.round(pred))]

  plt.imshow(img)
  plt.tile(f"Prediction:{pred_class}")
  plt.axis(False);
