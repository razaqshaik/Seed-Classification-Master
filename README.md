# Seed Classification using Pre-trained Deep Learning Models

A deep learning approach for seed classification using transfer learning with multiple pre-trained convolutional neural networks for comparative analysis.

## 📋 Overview

This project implements seed classification using a two-stage approach: feature extraction from pre-trained CNN models followed by custom classification layers. The study compares the performance of approximately 9 different pre-trained models including DenseNet121, InceptionV3, VGG19, ResNet, NASNet, MobileNet, and others.

## 🏗️ Methodology

### Two-Stage Approach
1. **Feature Extraction**: Extract features using pre-trained CNN models (without top classification layers)
2. **Classification**: Train custom dense neural networks on the extracted features

### Pre-trained Models Evaluated
- **DenseNet121** 
- **InceptionV3**
- **VGG19** 
- **ResNet**
- **NASNet**
- **MobileNet**
- **Additional models** (approximately 9 total for comprehensive comparison)

## 📊 Dataset Requirements

### Data Structure
Your dataset should be organized in the following structure:
```
your_dataset_folder/
├── train/
│   ├── seed_class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── seed_class_2/
│   ├── seed_class_3/
│   ├── seed_class_4/
│   └── seed_class_5/
└── test/
    ├── seed_class_1/
    ├── seed_class_2/
    ├── seed_class_3/
    ├── seed_class_4/
    └── seed_class_5/
```

### Dataset Specifications
- **Training Samples**: 4,000 images total
- **Testing Samples**: 1,000 images total
- **Image Requirements**: Any format (JPG, PNG, etc.)
- **Number of Classes**: 5 seed types
- **Recommended**: 800 images per class for training, 200 per class for testing

## 🚀 Setup Instructions

### Prerequisites
Install required libraries:
```
tensorflow
keras
numpy
scikit-learn
joblib
matplotlib
pillow
```

### Dataset Preparation

1. **Organize your custom dataset** in the folder structure shown above
2. **Update dataset paths** in the code:
   - Replace `train_dir` with your training folder path
   - Replace `test_dir` with your testing folder path
3. **Data splitting**: If you have a single dataset folder, you need to split it into train/test sets manually or programmatically
4. **Image format**: Ensure all images are in standard formats (JPG, PNG, etc.)

### Important Configuration Steps

#### 1. Dataset Path Configuration
- **Specify your dataset path**: Update the `train_dir` and `test_dir` variables to point to your custom dataset location
- **Example**: If your dataset is in `/home/user/seed_dataset/`, set paths accordingly

#### 2. Data Splitting Requirements
- **Manual Split**: Organize your data into separate train and test folders before running the code
- **Train/Test Ratio**: Recommended 80:20 split (4000 train, 1000 test images)
- **Class Balance**: Ensure each class has adequate representation in both train and test sets

#### 3. Model Parameters to Adjust
- **Sample Count**: Modify `sample_count` parameter based on your actual dataset size
- **Batch Size**: Adjust `batch_size` (default 32) based on your system memory
- **Number of Classes**: Update the final Dense layer if you have different number of seed classes

## 🛠️ Technical Implementation

### Feature Extraction Process
- DenseNet121 pre-trained model loaded without top classification layers
- Input images resized to 224×224×3 pixels
- Features extracted using global average pooling from DenseNet121
- Single feature extraction process with batch-wise processing for memory efficiency

### Classification Architecture Comparison
- **Input Layer**: 1024-dimensional feature vectors from DenseNet121
- **Multiple Classifier Designs**: Different architectures inspired by various models (InceptionV3, VGG19, ResNet, NASNet, MobileNet, etc.)
- **Hidden Layers**: Various combinations of Dense layers with ReLU activation, BatchNormalization, and Dropout
- **Output Layer**: Softmax activation for multi-class classification (5 seed classes)
- **Optimization**: Adam optimizer with categorical crossentropy loss

## 🔧 Workflow Steps

### Step 1: Data Preparation
- Organize your custom seed dataset in the required folder structure
- Ensure proper train/test split
- Verify image formats and quality

### Step 2: Feature Extraction
- Load DenseNet121 pre-trained model for feature extraction
- Extract features from your custom dataset using DenseNet121
- Save extracted features as single joblib file (`densenet_features.joblib`)

### Step 3: Model Training and Saving
- Load DenseNet121 extracted features from joblib file
- Train multiple different classifiers (InceptionV3-style, VGG19-style, ResNet-style, etc.) on same features
- Save each trained model as `.keras` file in their respective folders
- Apply different classification architectures to same feature set for comparison

### Step 4: Evaluation and Comparison
- Evaluate each model on test set
- Compare performance metrics across all models
- Analyze results for best performing architecture

## 📈 Experimental Design

### Feature Extraction Strategy
- **Single Feature Extraction**: Features extracted only using DenseNet121 model
- **Feature Caching**: DenseNet121 features saved as single `.joblib` file for reuse
- **Consistent Preprocessing**: Same image preprocessing applied once
- **Batch Processing**: Memory-efficient feature extraction from DenseNet121

### Comparative Analysis Approach
- DenseNet121 features used as input for all classifier models
- Multiple classifier architectures trained on same DenseNet121 features
- Identical train/test splits and features for fair comparison
- Different classification approaches (InceptionV3-style, VGG19-style, ResNet-style, etc.) applied to same feature set

## ⚠️ Important Notes

### Before Running the Code
1. **Update file paths** to match your custom dataset location
2. **Verify data split** - ensure you have separate train and test folders
3. **Check dataset size** - adjust sample counts in the code accordingly
4. **Validate image formats** - ensure all images can be loaded properly

### Memory Considerations
- **Large datasets**: Consider processing in smaller batches during feature extraction
- **Feature storage**: Single joblib file can be large for big datasets
- **GPU memory**: Monitor GPU usage during DenseNet121 feature extraction
- **Model storage**: Each trained classifier saved separately as `.keras` files for easy access

### Customization Requirements
- **Number of classes**: Modify final Dense layer output units in all classifier architectures
- **Dataset size**: Update sample_count parameters for feature extraction
- **Model architecture**: Adjust hidden layer sizes for different classifiers if needed
- **Training parameters**: Tune epochs, learning rate, etc. for each model
- **Model saving**: Ensure proper folder structure for saving each trained `.keras` model

## 🎯 Expected Outputs

### Generated Files
- Feature files (`.joblib`) for each pre-trained model
- Trained classifier models for each architecture
- Performance metrics and comparison results

### Analysis Results
- Accuracy comparison across all models
- Training time and computational efficiency analysis
- Model performance on your specific seed types
- Best performing architecture recommendation

## 🔬 Research Applications

- **Agricultural Technology**: Custom seed variety identification
- **Transfer Learning Research**: Architecture comparison for domain-specific tasks
- **Computer Vision**: Performance analysis on specialized datasets

---

*Configure your custom dataset paths and ensure proper data splitting before running the experiments.*

## 🚀 Deployment: Flask Web Application

After completing the model comparison and selecting the best performing classifier, the chosen model can be deployed as a web application for real-time seed classification.

### Flask Application Overview

The deployment involves creating a Flask web application that:
- Loads the best performing trained model (`.keras` file)
- Uses DenseNet121 for feature extraction from uploaded images
- Provides predictions through a user-friendly web interface
- Returns classification results with confidence scores

### Application Architecture

#### Backend Components
- **Flask Server**: Handles HTTP requests and file uploads
- **Feature Extraction**: DenseNet121 model for preprocessing images
- **Classification**: Best performing trained classifier model
- **Image Processing**: Handles image upload, validation, and preprocessing

#### Frontend Interface
- **File Upload**: Simple interface for uploading seed images
- **Results Display**: Shows predicted class and confidence scores
- **Error Handling**: User-friendly error messages for invalid inputs

### Key Features

#### Model Integration
- **DenseNet121 Feature Extractor**: Same preprocessing pipeline as training
- **Trained Classifier**: Loads the best performing `.keras` model from comparison study
- **Consistent Pipeline**: Identical feature extraction and classification process

#### Image Processing
- **Supported Formats**: PNG, JPG, JPEG image files
- **File Size Limit**: 16MB maximum upload size
- **Image Preprocessing**: Automatic resizing to 224×224 pixels
- **Security**: Secure filename handling and file validation

#### Prediction Output
- **Primary Prediction**: Most likely seed class with confidence score
- **All Probabilities**: Complete probability distribution across all classes
- **Class Labels**: Human-readable class names (Broken, Immature, Intact, Skin-Damaged, Spotted)

### Setup Requirements

#### Dependencies
```
flask
tensorflow
keras
numpy
pillow
werkzeug
```

#### File Structure
```
flask_app/
├── app.py                 # Main Flask application
├── inception_v3.keras     # Best performing trained model
├── templates/
│   └── index.html        # Frontend interface
├── uploads/              # Temporary image storage
└── static/               # CSS/JS files (optional)
```

#### Configuration Steps
1. **Model Selection**: Choose best performing model from comparison study
2. **Model Path**: Update model loading path in Flask app
3. **Class Labels**: Verify class names match your dataset classes
4. **Upload Directory**: Ensure upload folder exists and has write permissions

### Deployment Workflow

#### Step 1: Model Selection
- Analyze comparison results from all trained models
- Select model with highest accuracy or best performance metrics
- Copy the chosen `.keras` file to Flask application directory

#### Step 2: Application Setup
- Install required Flask dependencies
- Configure upload folder and file size limits
- Set allowed file extensions for image uploads
- Update class labels to match your seed types

#### Step 3: Feature Pipeline
- Load DenseNet121 model for feature extraction (same as training)
- Load selected trained classifier model
- Implement prediction pipeline matching training preprocessing

#### Step 4: Web Interface
- Create HTML template for file upload
- Implement JavaScript for handling responses
- Add error handling and user feedback
- Style interface for better user experience

### Usage Instructions

#### Running the Application
1. Start Flask development server
2. Navigate to web interface in browser
3. Upload seed image through file selector
4. View prediction results with confidence scores

#### Prediction Process
1. **Image Upload**: User selects and uploads seed image
2. **Preprocessing**: Image resized and normalized
3. **Feature Extraction**: DenseNet121 extracts features
4. **Classification**: Trained model predicts seed class
5. **Results**: Class label and confidence returned to user

### Production Considerations

#### Performance Optimization
- **Model Caching**: Load models once at startup
- **Batch Processing**: Handle multiple simultaneous requests
- **File Cleanup**: Remove uploaded files after processing
- **Error Logging**: Comprehensive error tracking and logging

#### Security Measures
- **File Validation**: Strict file type and size checking
- **Secure Uploads**: Prevent malicious file uploads
- **Input Sanitization**: Clean and validate all user inputs
- **Access Control**: Consider authentication for production use

#### Scalability Options
- **Docker Deployment**: Containerize application for easy deployment
- **Cloud Hosting**: Deploy on cloud platforms (AWS, GCP, Azure)
- **Load Balancing**: Handle multiple concurrent users
- **Database Integration**: Store prediction history and analytics

### Expected Results

#### Web Interface Features
- **Simple Upload**: Drag-and-drop or click to upload images
- **Real-time Results**: Immediate classification feedback
- **Confidence Scores**: Numerical confidence for predictions
- **All Classes**: Complete probability distribution display

#### Classification Output
- **Primary Class**: Most likely seed type
- **Confidence Level**: Prediction certainty percentage
- **Alternative Classes**: Other possible classifications with probabilities
- **Processing Time**: Quick response for real-time use

This deployment phase transforms your research comparison into a practical tool for automated seed classification, making the technology accessible to end users through an intuitive web interface.
