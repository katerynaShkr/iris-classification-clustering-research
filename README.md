# Research of Classification and Clustering Methods on the Iris Dataset

## 📌 Project Overview

This project focuses on the research and practical implementation of machine learning methods for data classification and clustering using the **Iris dataset**.

The project compares several machine learning approaches:

- **Classification using the Perceptron algorithm**
- **Agglomerative Hierarchical Clustering**
- **K-Means Clustering**

The main goal is to analyze the behavior of different machine learning algorithms, evaluate their performance, and compare supervised and unsupervised learning approaches.

---

## 🧠 Implemented Methods

### 1. Perceptron Classification

A linear classifier based on the classical Perceptron algorithm was implemented.

Features:

- training of weight vectors;
- bias implementation;
- weight updates using the Rosenblatt learning rule;
- visualization of the decision boundary;
- analysis of classification results.

File:

```text
task1_perceptron_iris.py
```

---

### 2. Agglomerative Clustering

An implementation of hierarchical clustering based on gradual merging of similar objects.

Features:

- bottom-up cluster formation;
- hierarchical data grouping;
- analysis of cluster structure;
- comparison with original Iris classes.

File:

```text
task2_agglomerative_clustering_iris.py
```

---

### 3. K-Means Clustering

A centroid-based clustering algorithm was implemented to group data points according to similarity.

Features:

- initialization of cluster centers;
- iterative centroid optimization;
- assigning objects to the nearest cluster;
- evaluation of clustering quality.

File:

```text
task3_kmeans_iris.py
```

---

## 📊 Dataset Description

The project uses the classic **Iris dataset by Ronald Fisher**.

Dataset characteristics:

- **150 samples**
- **3 flower classes:**
  - Iris setosa
  - Iris versicolor
  - Iris virginica

Each sample contains four numerical features:

- sepal length;
- sepal width;
- petal length;
- petal width.

The dataset is widely used for demonstrating and comparing machine learning algorithms due to its clear structure and simplicity.

---

## 🛠 Technologies

The project was developed using:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## ⚙️ Installation and Running

### 1. Clone the repository

```bash
git clone https://github.com/username/repository-name.git
```

Navigate to the project directory:

```bash
cd course_work
```

---

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

### 3. Run experiments

#### Perceptron classification:

```bash
python task1_perceptron_iris.py
```

#### Agglomerative clustering:

```bash
python task2_agglomerative_clustering_iris.py
```

#### K-Means clustering:

```bash
python task3_kmeans_iris.py
```

---

## 📈 Research Results

The project includes:

- dataset exploration and preprocessing;
- implementation of supervised learning classification;
- implementation of unsupervised clustering algorithms;
- visualization of obtained results;
- comparison of different machine learning approaches.

The quality of models is evaluated using appropriate classification and clustering metrics.

---

## 📂 Project Structure

```text
course_work/
│
├── task1_perceptron_iris.py
├── task2_agglomerative_clustering_iris.py
├── task3_kmeans_iris.py
│
└── README.md
```

---

## 🎓 Academic Information

Course:

**Numerical Methods and Machine Learning Technologies**

Specialty:

**126 — Information Systems and Technologies**

Project topic:

**"Research of Classification and Clustering Methods on the Example of the Iris Dataset"**

---

## 👤 Author

Student of Information Systems and Technologies  
National University "Odesa Polytechnic"

KATERYNA SHKURYNA