import numpy as np

from data_loader import get_train_and_test_datasets, load_images
from models.nn_classifier import NNClassifierVectorized


def main():
    dog_images, cat_images = load_images()
    x_train, x_test, y_train, y_test = get_train_and_test_datasets(
        dog_images, cat_images
    )
    classifier = NNClassifierVectorized()
    classifier.train(x_train, y_train)
    predicted_labels = [classifier.predict(test_image)[0] for test_image in x_test]
    print(predicted_labels[:10])

    y_test = np.array(y_test)
    predicted_labels = np.array(predicted_labels)

    accuracy = (y_test == predicted_labels).mean()
    print('Accuracy: {}%'.format(100*accuracy))

if __name__ == "__main__":
    main()
