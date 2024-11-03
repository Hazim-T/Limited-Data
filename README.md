# Image Classification with Transfer Learning on Sparse Datasets

This project applies transfer learning and custom CNN models using TensorFlow to classify images of different categories from two datasets: **Flowers** and **Plant Health**. 
This work demonstrates how transfer learning can be essential in areas with limited data, such as medical fields, by leveraging pre-trained models like VGG16 to improve classification performance.

## Project Structure

- **data/flowers**: Contains the "Flowers" dataset.
- **data/plant_health**: Contains the "Plant Health" dataset, with `Train` and `Test` subdirectories.
- **utils.py**: Utility functions for image visualization and predictions.
- **notebooks**: Includes code for model training and evaluation for each dataset.

## Results

Training was done on a baseline model with no prior weights and modified to find good accuracy, and on a model based on VGG16 with fine-tuning applied after initial training. 

The biggest faced issue was overfitting as the dataset was too small for the model to not memorize it. After experimenting, it seems that data augmentation and dropout 
used in conjunction with kernel regularization tend to most of the overfitting issues. Batch normalization was experimented with too, but seemed to give very varying results. Callbacks seem 
to be a necessity as well in limited data scenarios.

The selection of the pretrained model to use is also crucial. The dataset on which it was trained should be relatively close to the data it will be applied to. In our case, the VGG16 was a perfect fit
for the demonstration since it was trained on the imagenet dataset, which ports perfectly for plants which was the main focus in the testing here.

The most important takeaway from this experiment is that when working with a small dataset without transfer learning, trained models varied widely in results and contained quite a bit of randomness based
on how the data was shuffled and inputted. Transfer learning showed more stable results after attempting the training loop multiple times were loss and accuracy variance was minimal.

## Conclusion

Transfer learning applied properly is the core solution to any scenario with limited data. Any scenario that would prove difficult to collect data in would benefit greatly from using transfer learning.
So it is important to create a multitude of pretrained models varying in fields and data they were trained on to port them easily to other applications. I strongly believe more time and effort should
go into creating datasets for specializations, with an array of pretrained models on them to promote usability for transfer learning.

