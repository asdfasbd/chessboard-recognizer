# import tensorflow as tf
# import keras

# model = keras.layers.TFSMLayer(
#     "/home/m/chess/chessboard-recognizer/nn/model.tf/",
#     call_endpoint="serving_default"
# )
# model.save("./chessboard-recognizer/nn/model.keras")

import tensorflow as tf

model = tf.saved_model.load("/home/m/chess/chessboard-recognizer/nn/model.tf/")
inference_func = model.signatures["serving_default"]