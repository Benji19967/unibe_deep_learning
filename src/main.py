from data_loader import get_train_and_test_datasets, load_images
from models.nn_classifier import NNClassifierVectorized


def main():
    dog_images, cat_images = load_images()
    x_train, x_test, y_train, y_test = get_train_and_test_datasets(
        dog_images, cat_images
    )
    classifier = NNClassifierVectorized()
    classifier.train(x_train, y_train)
    predicted_label, min_distance, closest_image = classifier.predict(x_test[0])
    print(predicted_label)


if __name__ == "__main__":
    main()
