## Surgical Action Triplet Post Challenge Phase(Weakl Supervised Learning)

### Challenge Background  
This project is inspired by the [CholecTriplet 2022 Challenge](https://cholectriplet2022.grand-challenge.org/cholectriplet2022/) and aims to contribute to the advancement of fine-grained surgical activity modeling in computer-assisted surgery.  
For further details, refer to the official [CholecTriplet GitHub repository](https://github.com/CAMMA-public/cholectriplet2022).

Formalizing surgical activities as triplets of the used **instruments**, **actions performed**, and **target anatomies acted upon** provides a better understanding of surgical workflows. Automatic recognition of these triplet activities directly from surgical videos has the potential to revolutionize intra-operative decision support systems, enhancing safety and efficiency in the operating room (OR).

---

## Objective  
The challenge involves developing algorithms to:  
1. **Localize Instruments and Targets**: Identify the spatial regions of likelihood for instruments and targets within video frames.  
2. **Recognize Action Triplets**: Predict triplets of the form `{instrument, verb, target}` for each video frame.  
3. **Associate Triplets with Bounding Boxes**: Link action triplets to their corresponding localized bounding boxes.  

This is a weakly supervised learning problem where spatial annotations are not provided during training.

---

## Dataset  
The dataset used for this challenge is a subset of the **CholecT50** endoscopic video dataset, which is annotated with action triplet labels.  
- **Dataset structure**:  
  ```plaintext
  |──CholecT50
    ├───videos
    │   ├───VID01
    │   │   ├───000000.png
    │   │   ├───
    │   │   └───N.png
    │   ├───
    │   └───VIDN
    │       ├───000000.png
    │       ├───
    │       └───N.png
    ├───labels
    │   ├───VID01.json
    │   ├───
    │   └───VIDNN.json
    ├───label_mapping.txt        
    ├───LICENSE
    └───README.md
## Methodology

### 1. **Model Selection**

- **ResNet-50**: A pre-trained ResNet-50 model is used as the backbone for the task of tool count prediction. The model is fine-tuned to adapt to the specific problem of surgical tool detection by replacing the final fully connected layer to output six units, corresponding to six surgical tool classes. ResNet-50 is known for its deep architecture, which helps capture complex visual features and achieve high accuracy on medical imaging tasks.

- **Multi-task Model**: A multi-task learning approach is adopted to simultaneously predict tool counts and other related tasks, such as tool localization or classification. This model shares the feature extraction layers (e.g., the ResNet-50 backbone) but branches into separate heads for each task. Multi-task learning improves generalization by leveraging shared knowledge between related tasks, especially in scenarios where training data might be limited.

### 2. **Grad-CAM (Gradient-weighted Class Activation Mapping)**

- **Grad-CAM**: To visualize the regions of the image that are most important for the model’s predictions, Grad-CAM is applied. This technique generates class activation maps by using gradients of the target class flowing into the final convolutional layer. These maps highlight the most relevant areas in the input image that the model uses to make predictions. Grad-CAM helps interpret and validate the decision-making process of the model, especially in medical applications where understanding model behavior is crucial.

### 3. **Training and Evaluation**

- **Training**: The model is trained using the Adam optimizer with a learning rate of 0.0001 and a Mean Squared Error (MSE) loss function to predict the count of tools per frame. The model undergoes training for a limited number of steps to avoid overfitting and to quickly evaluate performance.
  
- **Evaluation**: After training, the model is evaluated using a separate validation dataset to ensure its generalization ability. The validation loss is tracked to monitor overfitting.

