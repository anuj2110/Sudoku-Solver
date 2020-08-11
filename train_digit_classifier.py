from main_module.models.sudoku_net import SudokuNet
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-m","--model",required=True,help="path to the output model file")

args = vars(ap.parse_args())

INIT_LR = 1e-3
EPOCHS = 10
BS = 128

print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

print("[INFO] loading and compiling network...")
model = SudokuNet.build(28,28,1,10)
opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(
	trainData, trainLabels,
	validation_data=(testData, testLabels),
	batch_size=BS,
	epochs=EPOCHS,
	verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))


print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")

