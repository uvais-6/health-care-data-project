from dataset import load_data
from model import create_model

X_train, X_test, y_train, y_test = load_data()
model = create_model()

model.fit(X_train, y_train, epochs=5)
loss, acc = model.evaluate(X_test, y_test)

print("Accuracy:", acc)
