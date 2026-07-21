# SkinSync
SkinSync is an AI-powered classification system designed to analyze skin conditions and offer personalizated skincare recommendations. Powered by Google’s MobileNetV2 architecture, the system leverages transfer learning and deep convolutional neural networks (CNNs) to perform accurate, real-time image classification.
## Currently Supported Analyses:
- Acne
- Skin type (oily/dry/normal)
- More hopefully soon
## Collaborators
- Krishanu Tandle
- Shrihan Danam
# Disclaimer
**SkinSync is an educational demonstration project and is not a certified medical device.** This tool does not provide medical advice, diagnosis, or treatment. Always consult a board-certified dermatologist or licensed healthcare professional for medical concerns or before starting a new skincare regimen.

## Acknowledgements & Citations
### Dataset
- **Image Dataset for SkinDiseases+Dry+Oily+NormalSkin**: Available on [Kaggle](https://www.kaggle.com/datasets/sd20co001/image-dataset-for-skindiseases-dry-oily-normalskin).
### Model Architecture
- **Backbone:** This project utilizes the **MobileNetV2** architecture pre-trained on ImageNet via PyTorch (`torchvision.models.mobilenet_v2`).
- **Original Paper:** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4510–4520).
