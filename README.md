# Bayesian Neural Network for Recommendation with Thompson Sampling

## 1. Project Overview

This project implements a sophisticated recommendation system that leverages a Custom Bayesian Neural Network (BNN) creating an approximation of Thompson Sampling. This approach allows the model to not only make predictions but also to quantify its uncertainty about those predictions. Thompson Sampling then uses this uncertainty to intelligently balance the exploration-exploitation trade-off, which is a powerful strategy in recommendation contexts.

The primary goal is to generate a set of recommendations for a given set of users. The final output is provided in both `.csv` and `.xlsx` formats.

## 2. Methodology

This project employs a practical and powerful approach to recommendation by approximating a Bayesian Neural Network (BNN) using **Monte Carlo (MC) Dropout**. This technique allows us to simulate Thompson Sampling for effective exploration-exploitation in our recommendations.

*   **Bayesian Neural Network via MC Dropout:** Instead of implementing a full, computationally expensive BNN, we approximate one using MC Dropout. In a standard neural network, dropout is a regularization technique used only during training. Here, we keep the dropout layers **active during inference (evaluation)**. Each time we pass the same input through the network, the dropout layers randomly "turn off" different neurons. This results in slightly different output predictions for each pass, creating a distribution of outcomes. This distribution is a proxy for the model's uncertainty, which is the key characteristic of a BNN.

*   **Approximating Thompson Sampling:** Thompson Sampling is a probabilistic algorithm for balancing the exploration-exploitation trade-off. It works by maintaining a belief (a probability distribution) about the potential reward of each possible action (or "arm"). To make a decision, it samples a potential reward value from each arm's distribution and chooses the arm with the highest sampled value. This elegantly allows it to explore arms with high uncertainty and high potential, while continuing to exploit arms it is confident are good.

    **Connection to our Model:** We implement an efficient approximation of this. A single forward pass through our `BanditBNN` model with dropout enabled (`model.train()` mode) is equivalent to drawing one sample from the approximate posterior distribution over rewards. The output scores for each item represent this single sample. By selecting the items with the highest scores from this pass, we are performing the core action of Thompson Sampling: choosing the best option based on one sample from our belief distribution. This allows the model to balance recommending items it knows are good (exploitation) with trying items it is uncertain but optimistic about (exploration).

## 3. Repository Contents

*   `Thompsom_BNN.ipynb`: This Jupyter Notebook contains the complete workflow for training the BNN model from scratch. It includes data loading, preprocessing, model architecture definition, the training loop, and validation. This notebook is provided for transparency and reproducibility of the model training process but is not required to generate the final results.
*   `evaluate_BNN.ipynb`: **This is the main notebook for generating the final recommendations.** It loads the pre-trained model (`bandit_bnn_model_final.pth`), processes the evaluation dataset, and generates the final submission files.
*   `bandit_bnn_model_final.pth`: This is the serialized, pre-trained PyTorch model file. It contains the learned distributions for the model weights and is ready for inference.
*   `submission.csv`: The final generated recommendations in CSV format.
*   `submission.xlsx`: The final generated recommendations in Excel format.

## 4. Setup and Installation

It is recommended to use a virtual environment to manage dependencies.

### Step 1: Create a Virtual Environment (Optional but Recommended)

```sh
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

All required libraries are listed in the `requirements.txt` file. You can install them all at once using `pip`:

```sh
pip install -r requirements.txt
```

The dependencies used are:
*   **pandas**: For data manipulation and reading CSV files.
*   **numpy**: For numerical operations.
*   **tqdm**: For displaying progress bars.
*   **torch**: The core deep learning framework.
*   **scikit-learn**: For utility functions like one-hot encoding and data splitting.
*   **jupyter**: To run the `.ipynb` notebooks.
*   **openpyxl**: Required by pandas to
```

## 5. How to Generate Recommendations

To generate the results, follow these steps:

1.  Ensure you have completed the setup and installation steps above.
2.  Launch Jupyter Notebook or Jupyter Lab:
    ```sh
    jupyter notebook
    ```
3.  From the Jupyter interface in your browser, open the `evaluate_BNN.ipynb` file.
4.  Run all the cells in the notebook by selecting "Cell" -> "Run All" from the menu bar.

## 6. Output

Upon successful execution of the `evaluate_BNN.ipynb` notebook, two new files will be created (or overwritten) in the same directory:

*   `submission.csv`
*   `submission.xlsx`

These files contain the final recommendations generated by the model for the evaluation dataset.